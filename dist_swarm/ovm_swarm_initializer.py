import sys
import traceback

from dist_swarm.db_bridge.rds_bridge import RDSCursor
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import os
import boto3
import grpc
import pickle
import json
import logging
import numpy as np
import threading
import concurrent
import psycopg2
import tensorflow.keras as keras
from pathlib import Path, PurePath
from pandas import read_pickle, read_csv
from time import gmtime, strftime

import data_process as dp
from models import get_model
from get_dataset import get_dataset
from get_optimizer import get_optimizer
from get_device import get_device_class
import models as custom_models
from dynamo_db import DEVICE_ID, ENCOUNTER_HISTORY, GOAL_DIST, HOSTNAME, IS_PROCESSED, LOCAL_DIST, MODEL_INFO, ORIG_ENC_IDX, TASK_ID, TOTAL_ENC_IDX,\
    DATA_INDICES, EVAL_HIST_LOSS, EVAL_HIST_METRIC, ENC_IDX, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, \
    MODEL_INFO, WC_TIMESTAMPS, WORKER_ADDED, WORKER_ID, WORKER_STATUS, WORKER_HISTORY, WORKER_CREATED, WTIMESTAMP, ACTION_TYPE
from grpc_components.status import STOPPED
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc
from aws_settings import REGION
from get_model import get_model_fn
from device.check_device_type import is_hetero_strategy
from ovm_utils.device_storing_util import save_device

client = boto3.client('dynamodb', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)

class OVMSwarmInitializer():
    def initialize(self, config_file, create_tables=False, init_worker=True) -> None:
        self._config_db(config_file, create_tables, init_worker)
        
    def _delete_all_items_on_table(self, table_name):
        try:
            table = dynamodb.Table(table_name)
            #get the table keys
            tableKeyNames = [key.get("AttributeName") for key in table.key_schema]

            #Only retrieve the keys for each item in the table (minimize data transfer)
            projectionExpression = ", ".join('#' + key for key in tableKeyNames)
            expressionAttrNames = {'#'+key: key for key in tableKeyNames}

            counter = 0
            page = table.scan(ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames)
            with table.batch_writer() as batch:
                while page["Count"] > 0:
                    counter += page["Count"]
                    # Delete items in batches
                    for itemKeys in page["Items"]:
                        batch.delete_item(Key=itemKeys)
                    # Fetch the next page
                    if 'LastEvaluatedKey' in page:
                        page = table.scan(
                            ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames,
                            ExclusiveStartKey=page['LastEvaluatedKey'])
                    else:
                        break
            # print(f"Deleted {counter}")
        except:
            # print("new table; nothing to delete")
            pass 

    def _create_finished_tasks_table(self, tag):
        self._create_table(TASK_ID, tag+'-finished-tasks', 20, 20, IS_PROCESSED)

    def _create_db_state_table(self, tag):
        self._create_table(DEVICE_ID, tag, 20, 20)

    def _get_worker_state_table_name(self):
        return self.worker_namespace + '-worker-state'

    # def _create_worker_state_table(self):
    #     self._create_table(WORKER_ID, self._get_worker_state_table_name(), 40, 300, clear_table=False)

    def _create_table(self, key, table_name, 
                      read_cap_units=100, write_cap_units=100, secondary_index=None,
                      clear_table=True):  
        """
        stores the state of the worker nodes
        """

        params = {
            'TableName':table_name,
                # Declare your Primary Key in the KeySchema argument
            'KeySchema':[
                {
                    "AttributeName": key,
                    "KeyType": "HASH"
                }
            ],
            # Any attributes used in KeySchema or Indexes must be declared in AttributeDefinitions
            'AttributeDefinitions': [
                {
                    "AttributeName": key,
                    "AttributeType": "N"
                }
            ],
            # ProvisionedThroughput controls the amount of data you can read or write to DynamoDB per second.
            # You can control read and write capacity independently.
            'ProvisionedThroughput': {
                "ReadCapacityUnits": read_cap_units,
                "WriteCapacityUnits": write_cap_units
            }
        }

        # if secondary_index is not None:
        #     params['GlobalSecondaryIndexes'] = [{
        #         'IndexName': secondary_index,
        #         'KeySchema': [
        #             {
        #                 'AttributeName': secondary_index,
        #                 'keyType': "HASH"
        #             }
        #         ]
        #     }]

        try:
            resp = client.create_table(
                **params
            )
            print("Table created successfully. Syncing...")
            waiter = client.get_waiter('table_exists')
            waiter.wait(
                TableName=table_name,
                WaiterConfig={
                    'Delay': 5,
                    'MaxAttempts': 10
                }
            )
            print(f"Table {table_name} Successfully created")

        except Exception as e:
            pass
            print("Error creating table:")
            print(e)

        # # delete all the existing items in the db
        if clear_table:
            self._delete_all_items_on_table(table_name)

    def send_set_worker_state_request(self, swarm_name, worker_id):
        with grpc.insecure_channel(self.worker_id_to_ip[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
            stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
            worker_info = simulate_device_pb2.WorkerInfo(swarm_name=swarm_name, worker_namespace=self.worker_namespace, worker_id=worker_id,
                            rds_host=self.rds_endpoint, rds_dbname=self.rds_dbname, rds_user=self.rds_user, 
                            rds_password=self.rds_password, rds_table='k8s')
            status = stub.SetWorkerInfo.future(worker_info)
            res = status.result()
            logging.info(f"{self.worker_id_to_ip[worker_id]} set as {worker_id}")

    def _initialize_worker(self, tag, worker_id):
        # table = dynamodb.Table(self._get_worker_state_table_name())
        # with table.batch_writer() as batch:
        #     batch.put_item(Item={WORKER_ID: worker_id, WORKER_STATUS: STOPPED,
        #                        WORKER_HISTORY: [{WTIMESTAMP: strftime("%Y-%m-%d %H:%M:%S", gmtime()), ACTION_TYPE: WORKER_ADDED}]})
        set_number_thread = threading.Thread(target=self.send_set_worker_state_request, args=(tag, worker_id,))
        set_number_thread.start()
    
    def _config_db(self, config, create_tables=False, init_worker=True):
        # load config file
        self.config = config
        self.swarm_name = config['tag']
        tag = config['tag']
        self.swarm_name = tag
        swarm_config = config['swarm_config']
        self.worker_ips = config['worker_ips']
        self.worker_namespace = "deprecated"
        self.swarm_init_group = config['swarm_init_group']
        if 'bucket' in config:
            self.bucket = config['bucket']
        else:
            self.bucket = 'opfl-sim-models'

        # setup RDS cursor
        rds_config = config['rds_config']
        self.rds_config = rds_config
        self.rds_endpoint = rds_config['rds_endpoint']
        self.rds_user = rds_config['rds_user']
        self.rds_password = rds_config['rds_password']
        self.rds_dbname = rds_config['rds_dbname']

        self.tasks_table_name = self.swarm_name + '_finished_tasks'

        # @TODO create k8s table if not exist
        self.rds_cursor = RDSCursor(self.rds_endpoint, self.rds_dbname, self.rds_user, self.rds_password, 'k8s')
        self.rds_tasks_cursor = RDSCursor(self.rds_endpoint, self.rds_dbname, self.rds_user, self.rds_password, self.swarm_name+'_finished_tasks')
        if create_tables:
            self.rds_tasks_cursor.execute_sql(f'create table if not exists {self.tasks_table_name} ( \
                                                serial_id serial PRIMARY KEY, \
                                                task_id INTEGER NOT NULL, \
                                                is_processed BOOLEAN NOT NULL, \
                                                is_finished BOOLEAN NOT NULL, \
                                                is_timed_out BOOLEAN NOT NULL, \
                                                sim_time NUMERIC (10, 4), \
                                                wc_time NUMERIC (10, 4), \
                                                learner INTEGER NOT NULL, \
                                                neighbor VARCHAR (40), \
                                                sim_timestamp NUMERIC (14, 2), \
                                                wc_timestamp NUMERIC (14, 2), \
                                                loss NUMERIC (10, 4), \
                                                metric NUMERIC (10, 4), \
                                                enc_idx INTEGER, \
                                                undefined VARCHAR (20) \
                                                )')
            self.rds_tasks_cursor.clear_all()

        if config['dataset'] is not 'femnist':
            x_train, y_train_orig, x_test, y_test_orig = get_dataset(config['dataset'])
            num_classes = len(np.unique(y_train_orig))
        else:
            x_train, y_train_orig, x_test, y_test_orig = (0,0,0,0)
            num_classes = 0

        if create_tables:
            self._create_db_state_table(tag)

        enc_dataset_filename = self.config['device_config']['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)

        # initialize all devices and store their states, models, data, etc
        # store local data dist, goal dist, and training data indices in the table

        # first, init devices in device groups
        cur_id = 0
        global_device_config = config['device_config']
        if 'device_groups' in global_device_config:
            device_groups = global_device_config['device_groups']
            for dg in device_groups:
                self._create_and_save_device(cur_id, config, dg['device_config'],
                                             num_classes, x_train, y_train_orig,
                                             x_test, y_test_orig, enc_df)
                cur_id += 1

        print("creating device states")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for idnum in range(cur_id, cur_id + swarm_config['number_of_devices']):
                if not create_tables:
                    continue
                futures.append(executor.submit(self._create_and_save_device,
                                idnum, config, config['device_config'],
                                num_classes, x_train, y_train_orig,
                                x_test, y_test_orig, enc_df))

            concurrent.futures.wait(futures)
        print("created device states")

        # @TODO Check if devices are all there, if not, initialize again
        
        # configure worker tables
        # if create_tables:
        #     self._create_worker_state_table()
        self.worker_ip_to_id = {}
        self.worker_id_to_ip = {}
        for idx in range(len(self.worker_ips)):
            # find worker on RDS and insert if not exist
            self.rds_cursor.insert_record(['external_ip', 'worker_state'], [self.worker_ips[idx], STOPPED])
            worker_id_resp = self.rds_cursor.get_column('worker_id', 'external_ip', self.worker_ips[idx])
            self.worker_ip_to_id[self.worker_ips[idx]] = worker_id_resp[0][0]
            self.worker_id_to_ip[worker_id_resp[0][0]] = self.worker_ips[idx]
            if init_worker:
                self._initialize_worker(tag, worker_id_resp[0][0])
       
        # if create_tables:
        #     self._create_finished_tasks_table(tag)

    
    def _create_and_save_device(self, idnum, config, device_config,
                                num_classes, x_train, y_train_orig,
                                x_test, y_test_orig, enc_df):
        try:
            swarm_config = config['swarm_config']
            if config['dataset'] != 'femnist':
                # pick data
                label_set = []
                goal_dist = {}
                local_dist = {}
                if "district-9" in config:
                    label_set.append(swarm_config["district-9"][idnum % len(swarm_config["district-9"])])
                else:
                    label_set = (np.random.choice(np.arange(num_classes), size=swarm_config['local_set_size'], replace=False)).tolist()
                
                for l in label_set:
                    local_dist[l] = (int) (swarm_config['local_data_size'] / len(label_set))

                labels_not_in_local_set = np.setdiff1d(np.arange(num_classes), np.array(label_set))
                label_set.extend((np.random.choice(labels_not_in_local_set, 
                                            size=swarm_config['goal_set_size'] - swarm_config['local_set_size'], replace=False)).tolist())
                for l in label_set:
                    goal_dist[l] = (int) (swarm_config['local_data_size'] / len(label_set))

                train_data_provider = dp.IndicedDataProvider(x_train, y_train_orig, local_dist)
                chosen_data_idx = train_data_provider.get_chosen()
            else:
                x_local, y_local_orig, _, _ = get_dataset('femnist', idnum)
                goal_dist = dict.fromkeys(range(62), 1)
                local_dist = dict.fromkeys(range(62), 1)


            table = dynamodb.Table(config['tag'])
            with table.batch_writer() as batch:
                batch.put_item(Item={DEVICE_ID: idnum, DEV_STATUS: STOPPED, TIMESTAMPS: [], WC_TIMESTAMPS: [],
                    GOAL_DIST: convert_to_map(goal_dist),
                    LOCAL_DIST: convert_to_map(local_dist), TOTAL_ENC_IDX: len(enc_df.index),
                    ENCOUNTER_HISTORY: [], EVAL_HIST_LOSS: [], EVAL_HIST_METRIC: [], ORIG_ENC_IDX: [],
                    ENC_IDX: -1, ERROR_TRACE: {}, HOSTNAME: 'N/A', MODEL_INFO: {}})
            

            ## initialize device 
            # get model and dataset
            model_fn = get_model(config['dataset'])

            self.device_class = get_device_class(device_config['device_strategy'])

            # bootstrap parameters
            if device_config['pretrained_model'] != "none":
                ext = device_config['pretrained_model'].split('.')[-1]
                pretrained_model_path = PurePath(os.path.dirname(__file__) +'/' + device_config['pretrained_model'])
                if ext == 'pickle':
                    with open(pretrained_model_path, 'rb') as handle:
                        init_weights = pickle.load(handle)
                else:
                    model = keras.models.load_model(pretrained_model_path, compile=False)
                    init_weights = model.get_weights()
            else:
                init_weights = None

            if config['dataset'] != 'femnist':
                test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, device_config['train_config']['test-data-per-label'])

                # get device info from dynamoDB
                chosen_list = chosen_data_idx
                goal_labels = goal_dist
                
                train_data_provider.set_chosen(chosen_list) 

                # prepare params for device
                x_local, y_local_orig = train_data_provider.fetch()
                
            else:
                train_data_provider = 0
                test_data_provider = dp.DummyTestDataProvider()
                goal_labels = [0]

            hyperparams = device_config['train_config']
            compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
            train_config = {'batch_size': hyperparams['batch-size'], 'shuffle': True}
            

            device = self.device_class(idnum,
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

            # save device model, dataset, and device object on S3 
            save_device(device, config['tag'], config['swarm_init_group'], -1, True, self.bucket)
        except:
            logging.error(f"{traceback.format_exc()}")

        logging.info(f"device {idnum} initialization complete")

def convert_to_map(dist):
    new_dist = {}
    for k in dist:
        new_dist[str(k)] = dist[k]
    return new_dist

def convert_from_map(m):
    dist = {}
    for k in m:
        dist[int(m)] = m[k]
    return dist

if __name__ == '__main__':
    dist_swarm = OVMSwarm('../configs/dist_swarm/controller_example.json', '')
