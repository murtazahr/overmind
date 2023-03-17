import numpy as np
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras import backend as K
import copy
import pickle
import data_process as dp
import datetime
import logging
import numpy as np
from tqdm import tqdm

from get_models import get_init_model_with_accuracy

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class DROppCLSwarm():
    def __init__(self, model_fn, opt_fn,
                 client_class,
                 num_clients, 
                 x_train, y_train, 
                 test_data_provider,
                 num_label_per_client,
                 num_req_label_per_client,
                 num_data_per_label_in_client,
                 enc_exp_config, hyperparams, from_swarm=None, log_callback=None):
        """
        enc_exp_config: dictionary for configuring encounter data based experiment
        [keys]
            data_file_name: pickle filename for pandas dataframe
            send_duration: how long does it take to send/receive the model?
            delegation_duration: how long does it take to run a single delegation
            max_delegations: maximum delegation rounds
        """
        self.run_times = 0
        self.last_run_time = 0
        self.test_data_provider = test_data_provider
        compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
        train_config = {'batch_size': hyperparams['batch-size'], 'shuffle': True}
        self.hyperparams = hyperparams
        self.log_callback = log_callback

        self.train_data_provider = dp.DataProvider(x_train, y_train)
        self.num_data_per_label_in_client = num_data_per_label_in_client

        local_data_labels = [0,1,2,3,4,5,6,7,8,9]
        target_labels = np.array(local_data_labels)
        tmp = [num_data_per_label_in_client] * len(local_data_labels)

        if from_swarm == None:
            self._clients = []
            for i in range(num_clients):
                x_train_client, y_train_client = dp.filter_data_by_labels_with_numbers(x_train, y_train, dict(zip(local_data_labels, tmp)))
                m = model_fn(size=enc_exp_config['client-sizes']['model-sizes'][i % 5])
                self._clients.append(client_class(i,
                                    model_fn,
                                    opt_fn,
                                    copy.deepcopy(m.get_weights()),
                                    x_train_client,
                                    y_train_client,
                                    self.train_data_provider,
                                    self.test_data_provider,
                                    target_labels,  # assume that required d.d == client d.d.
                                    compile_config,
                                    train_config,
                                    hyperparams))
                del m

        else:
            self._clients = []
            for i in range(num_clients):
                x_train_client, y_train_client = dp.filter_data_by_labels_with_numbers(x_train, y_train, dict(zip(local_data_labels, tmp)))
                m = model_fn(size=enc_exp_config['client-sizes']['model-sizes'][i % 5])
                self._clients.append(client_class(i,
                                    model_fn,
                                    opt_fn,
                                    copy.deepcopy(m.get_weights()),
                                    from_swarm._clients[i]._x_train,
                                    from_swarm._clients[i]._y_train_orig,
                                    from_swarm.train_data_provider,
                                    from_swarm.test_data_provider,
                                    list(from_swarm._clients[i]._desired_data_dist.keys()),  # assume that required d.d == client d.d.
                                    compile_config,
                                    train_config,
                                    hyperparams))
                del m
        
        self.hist = {} # history per client over time

        self.hist['clients'] = {}
        for i in range(num_clients):
            self.hist['clients'][i] = []

        self.hist['clients_unknown'] = {}
        for i in range(num_clients):
            self.hist['clients_unknown'][i] = []

        self.hist['clients_local'] = {}
        for i in range(num_clients):
            self.hist['clients_local'][i] = []
            
        self._config = enc_exp_config
        self.enc_df = pd.read_csv(self._config['encounter-data-file'])
        self.total_number_of_rows = self.enc_df.shape[0]

        self.hist['time_steps'] = [0]
        self.hist['loss_max'] = []
        self.hist['loss_min'] = []
        self.hist['total_rounds'] = 0  # total number of rounds
        self.hist['total_requests'] = 0  # total number of request of gradient computation
        self.hist['total_used_encs'] = 0  # used encounters among all encounters, even if only one of the devices requested computation of gradients
        self.hist['total_encs'] = 0  # total number of encounters
        self.hist['total_fr'] = 0
        self.hist['encounters_and_exchanges'] = []

    def _evaluate_all(self):
        #  run one local updates each first
        for i in range(len(self._clients)):
            hist = self._clients[i].eval()
            self.hist['clients'][i].append((0, hist, [])) # assume clients all start from the same init
            self.hist['loss_max'].append(hist[0])
            self.hist['loss_min'].append(hist[0])
        
        self.dropout_hist = {}
        self.quantization_hist = {}
        self.iteration_hist = {}
        for i in range(len(self._clients)):
            self.dropout_hist[i] = {}
            self.quantization_hist[i] = {}
            self.iteration_hist[i] = []

    def _initialize_last_times(self):
        self.last_end_time = {}
        for i in range(len(self._clients)):
            self.last_end_time[i] = 0
        self.last_data_update_time = {}
        for i in range(len(self._clients)):
            self.last_data_update_time[i] = 0

    def run(self, upto, allowOverlap=False):
        # stores the end time of the last encounter
        # this is to prevent one client exchanging with more than two
        # at the same time
        if self.run_times == 0:
            self._evaluate_all()
            self._initialize_last_times()
        self.run_times += 1

        print("running {} times".format(self.run_times))

        print("Start running simulation with {} indices".format(self.total_number_of_rows))
        start_time = datetime.datetime.now()
        # iterate encounters
        cur_t = 0 # current time
        end_t = 0
        cur_idx = 0

        for index, row in self.enc_df.iterrows():
            self.hist['total_encs'] += 1
            print(self.hist['total_encs'])
            if cur_idx > upto:
                break
            cur_idx += 1
            cur_t = row[TIME_START]
            end_t = row[TIME_END]
            duration = end_t - cur_t # time left
            # only pairs of clients can exchange in a place
            c1_idx = (int)(row[CLIENT1])
            c2_idx = (int)(row[CLIENT2])
            if c1_idx == c2_idx:
                continue
            if c1_idx >= len(self._clients) or c2_idx >= len(self._clients):
                continue
            c1 = self._clients[c1_idx]
            c2 = self._clients[c2_idx]
            if not allowOverlap and (self.last_end_time[c1_idx] > cur_t + self.last_run_time or self.last_end_time[c2_idx] > cur_t + self.last_run_time):
                continue  # client already occupied

            # assume both clients are fully occupied for delegation(so side-delegation in one time period)
            self.hist['total_used_encs'] += 1
            iter1, d_l_1, n1 = c1.hetero_delegate(c2, 1, duration)
            iter2, d_l_2, n2 = c2.hetero_delegate(c1, 1, duration)
            self.hist['total_requests'] += iter1 + iter2
            self.iteration_hist[c1_idx].append(iter1)
            self.iteration_hist[c2_idx].append(iter2)
            
            if iter1 != 0:
                self._put_hist(self.dropout_hist, c1_idx, d_l_1)
                self._put_hist(self.quantization_hist, c1_idx, n1)
                # if index != 0 and index % 100 == 0:
                #     hist = c1.eval()
                #     self.hist['clients'][c1_idx].append((self.last_end_time[c1_idx] + self.last_run_time, hist, 0))
            
            if iter2 != 0:
                self._put_hist(self.dropout_hist, c2_idx, d_l_2)
                self._put_hist(self.quantization_hist, c2_idx, n2)
                # if index != 0 and index % 100 == 0:
                #     hist = c2.eval()
                #     self.hist['clients'][c2_idx].append((self.last_end_time[c2_idx] + self.last_run_time, hist, 0))

            self.last_end_time[c1_idx] = cur_t + duration
            self.last_end_time[c2_idx] = cur_t + duration 

            self.hist['iteration_hist'] = self.iteration_hist
            self.hist['dropout_hist'] = self.dropout_hist
            self.hist['quantization_hist'] = self.quantization_hist 

            if (index != 0 and index % 100 == 0) or index == len(self.enc_df.index) - 1:
                tot_loss, tot_acc = self._get_tot_loss_and_acc()
                self.log_callback('[index {}]: tot_loss: {}, tot_acc: {}'.format(index, tot_loss, tot_acc))
                self.log_callback('[index {}]: tot_encs: {}, tot_reqs: {}'.format(index, self.hist['total_used_encs'], self.hist['total_requests']))
            
            if index != 0 and index % 500 == 0:
                elasped = datetime.datetime.now() - start_time
                rem = elasped / (index+1) * (self.total_number_of_rows-index-1)
                print("\n------------ index {} done ---".format(index), end='') 
                print("elasped time: {}".format(elasped), end='')
                print(" ----  remaining time: {}".format(rem))
                if self.log_callback != None:
                    self.log_callback('index {} done ---'.format(index))

            K.clear_session()

        # temp: evaluate only at last time to save simulation time
        for c in self._clients:
            hist = c.eval()
            self.hist['clients'][c._id_num].append((self.last_end_time[c._id_num], hist, 0))    
        
        self.last_run_time += end_t

    def _get_tot_loss_and_acc(self):
        tot_loss = 0
        tot_acc = 0
        
        for c in self._clients:
            tot_loss += self.hist['clients'][c._id_num][-1][1][0]
        for c in self._clients:
            tot_acc += self.hist['clients'][c._id_num][-1][1][1]

        tot_loss /= len(self._clients)
        tot_acc /= len(self._clients)

        return tot_loss, tot_acc

    def _put_hist(self, dic, client_idx, val):
        if val not in dic[client_idx]:
            dic[client_idx][val] = 0
        dic[client_idx][val] += 1

    def register_table(self, *args):
        print('no table used in this class')

    def delete_local_objects(self):
        self.log_callback = None
