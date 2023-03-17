import sys
sys.path.insert(0,'..')
import os
import time
from dist_swarm.db_bridge.model_in_db import ModelInDB
import socket
import tensorflow.keras as keras
import pickle
import threading
from pathlib import Path, PurePath
from cfg_utils import OUTPUT_FOLDER
from pandas import read_pickle, read_csv
import logging
import traceback

from models import get_model
from get_dataset import get_dataset
from get_optimizer import get_optimizer
from get_device import get_device_class
import data_process as dp
from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED
from grpc_components.simulate_device_pb2 import Status
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from device.check_device_type import is_hetero_strategy

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class OppCLDevice():
    ### gRPC methods
    def __init__(self, json_config):
        self.config = json_config
        self.device_config = json_config['device_config']
        
        try:
            self._initialize_model_in_db(self.config)
            logging.debug('device {} initializing'.format(self.device_config['id']))
            self.device_in_db = DeviceInDB(self.config['tag'], self.device_config['id'])
            self.device = self._initialize_device(self.config)
            self.device_in_db.update_status(IDLE)
        except Exception as e:
            self._handle_error(traceback.format_exc())

    def run(self):
        if self.device_in_db.get_status() == ERROR:
            logging.error('device is in error state')
            return self._str_to_status(ERROR)
        self.device_in_db.set_hostname(socket.gethostname())
        try:
            oppcl_thread = threading.Thread(target=self._start_oppcl, args=())
            oppcl_thread.start()
            return self._str_to_status(RUNNING)
        except Exception as e:
            return self._handle_error(traceback.format_exc())

    def _start_oppcl(self):
        try:
            enc_dataset_filename = self.device_config['encounter_config']['encounter_data_file']
            enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
            if enc_dataset_filename.split('.')[-1] == 'pickle':
                with open(enc_dataset_path,'rb') as pfile:
                    enc_df = read_pickle(pfile)
            else:
                with open(enc_dataset_path,'rb') as pfile:
                    enc_df = read_csv(pfile)
            last_end_time = 0
            last_run_time = 0
            self.device_in_db.update_status(RUNNING)

            self.hist_loss = []
            self.hist_metric = []
            self.timestamps = []
            self.other_device_cache = {}

            prev_sys_time = time.time()

            repeat = 1
            if 'repeat' in self.config['swarm_config']:
                repeat = int(self.config['swarm_config']['repeat'])
            
            for rep in range(repeat):
                for index, row in enc_df.iterrows():
                    # check status and exit if not running
                    self.device_in_db.fetch_status()
                    if self.device_in_db.get_status() != RUNNING:
                        raise RuntimeError('Process force stopped')

                    if (int)(row[CLIENT1]) == self.device._id_num:
                        other_id = (int)(row[CLIENT2])
                    elif (int)(row[CLIENT2]) == self.device._id_num:
                        other_id = (int)(row[CLIENT1])
                    else:
                        continue

                    if other_id == self.device._id_num or other_id >= self.config['swarm_config']['number_of_devices']:
                        continue
                    
                    if other_id not in self.other_device_cache:
                        other_device_in_db = DeviceInDB(self.config['tag'], other_id)
                        other_device_in_db.fetch_status() #TODO cache status. currently it is too read heavy

                        # get device info from dynamoDB and cache them 
                        self.other_device_cache[other_id] = {}
                        self.other_device_cache[other_id]['chosen_list']  = other_device_in_db.get_data_indices()
                        self.other_device_cache[other_id]['goal_labels'] = other_device_in_db.get_goal_labels()

                    other_chosen_list = self.other_device_cache[other_id]['chosen_list']
                    other_goal_labels = self.other_device_cache[other_id]['goal_labels']

                    other_train_data_provider = dp.IndicedDataProvider(self.x_train, self.y_train_orig, None)
                    other_test_data_provider = dp.StableTestDataProvider(self.x_test, self.y_test_orig, self.device_config['train_config']['test-data-per-label'])
                    other_train_data_provider.set_chosen(other_chosen_list)

                    other_x_local, other_y_local_orig = other_train_data_provider.fetch()

                    hyperparams = self.config['device_config']['train_config']
                    other_device = self.device_class(other_id,
                                                    None, 
                                                    None,
                                                    None,
                                                    other_x_local,
                                                    other_y_local_orig,
                                                    other_train_data_provider,
                                                    other_test_data_provider,
                                                    other_goal_labels,
                                                    None,
                                                    None,
                                                    hyperparams)
                    
                    if self.device.decide_delegation(other_device):
                        # calculate time
                        cur_t = row[TIME_START] + last_run_time
                        end_t = row[TIME_END] + last_run_time
                        time_left = end_t - cur_t
                        if last_end_time > cur_t:
                            continue
                        
                        if is_hetero_strategy(self.config['device_config']['device_strategy']):
                            self.device.hetero_delegate(other_device, 1, time_left)
                            last_end_time = end_t
                        else:
                            # determine available rounds of training and conduct OppCL
                            encounter_config = self.device_config['encounter_config']
                            model_send_time = self.device_config['model_size_in_bits'] / encounter_config['communication_rate']
                            computation_time = encounter_config['computation_time']
                            oppcl_time = 2 * model_send_time + computation_time
                            rounds = (int) ((time_left) / oppcl_time)
                            rounds = min(rounds, self.device_config['train_config']['max_rounds'])
                            if rounds < 1:
                                continue
                            for r in range(rounds):
                                self.device.delegate(other_device, 1, 1)
                            last_end_time = cur_t + rounds * oppcl_time

                        # evaluate
                        hist = self.device.eval()
                        self.hist_loss.append(hist[0])
                        self.hist_metric.append(hist[1])
                        self.timestamps.append(last_end_time)

                        # report eval to dynamoDB @TODO catch error
                        # logging.info('device: {}, index {}'.format(self.device._id_num, index))
                        # self.device_in_db.update_loss_and_metric(hist[0], hist[1], index)
                        cur_sys_time = time.time()
                        timediff = cur_sys_time - prev_sys_time
                        if timediff > 2:
                            self.device_in_db.update_loss_and_metric_in_bulk(self.hist_loss, self.hist_metric, index)
                            self.device_in_db.update_timestamps_in_bulk(self.timestamps)
                            prev_sys_time = time.time()
                        
                        # @TODO for sync device, upload model to S3 here
                    last_run_time = last_end_time
                        

            logging.info('device: {}: simulation complete.'.format(self.device._id_num))
            self.device_in_db.update_loss_and_metric_in_bulk(self.hist_loss, self.hist_metric, len(enc_df.index)-1)
            self.device_in_db.update_status(FINISHED)

        except Exception as e:
            print(e)
            return self._handle_error(traceback.format_exc())

    def _handle_error(self, e):
        logging.error('error: {}'.format(e))
        self.device_in_db.set_error(e)
        return self._str_to_status(ERROR)

    def _str_to_status(self, st):
        return Status(status=st)

    def _initialize_model_in_db(self, config):
        if not hasattr(self, 'config'):
            self.config = config

        # setup local path for storing model locally
        self.model_folder = OUTPUT_FOLDER + '/models'
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        # setup S3
        self.model_in_db = ModelInDB(config['tag'], config['device_config']['id'])

    def _initialize_device(self, config):
        # get model and dataset
        self.x_train, self.y_train_orig, self.x_test, self.y_test_orig = get_dataset(config['dataset'])
        model_fn = get_model(config['dataset'])

        self.device_class = get_device_class(config['device_config']['device_strategy'])

        # bootstrap parameters
        if config['device_config']['pretrained_model'] != "none":
            pretrained_model_path = PurePath(os.path.dirname(__file__) +'/' + config['device_config']['pretrained_model'])
            with open(pretrained_model_path, 'rb') as handle:
                init_weights = pickle.load(handle)
        else:
            init_weights = None

        train_data_provider = dp.IndicedDataProvider(self.x_train, self.y_train_orig, None)
        test_data_provider = dp.StableTestDataProvider(self.x_test, self.y_test_orig, config['device_config']['train_config']['test-data-per-label'])
        
        self.device_in_db.fetch_status()

        # get device info from dynamoDB
        chosen_list = self.device_in_db.get_data_indices()
        goal_labels = self.device_in_db.get_goal_labels()
        
        train_data_provider.set_chosen(chosen_list)

        # prepare params for device
        x_local, y_local_orig = train_data_provider.fetch()
        hyperparams = config['device_config']['train_config']
        compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
        train_config = {'batch_size': hyperparams['batch-size'], 'shuffle': True}

        device = self.device_class(config['device_config']['id'],
                                    model_fn, 
                                    get_optimizer(config['device_config']['train_config']['optimizer']),
                                    init_weights,
                                    x_local,
                                    y_local_orig,
                                    train_data_provider,
                                    test_data_provider,
                                    goal_labels,
                                    compile_config,
                                    train_config,
                                    hyperparams)

        if is_hetero_strategy(self.config['device_config']['device_strategy']):
            self.device_in_db.update_model_info('model_size: {}, device_power:{}'.format(device.model_size, device.device_power))

        return device
