from ast import parse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from timeit import default_timer as timer
import matplotlib
import copy
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import models as custom_models
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_opp_uci_dataset, get_svhn_dataset
from get_device import get_device_class
import device.exp_device
import pickle
import argparse
from droppcl_swarm import DROppCLSwarm
import data_process as dp
from swarm_utils import get_time
import data_process as dp
import boto3
from cfg_utils import setup_env, LOG_FOLDER, FIG_FOLDER, HIST_FOLDER
from pathlib import PurePath, Path
import logging
import sys

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

# S3 client and bucket
client = boto3.client('s3')
S3_BUCKET_NAME = 'opfl-sim-models'

def main():
    setup_env()

    # parse arguments
    parser = argparse.ArgumentParser(description='set params for simulation')
    parser.add_argument('--seed', dest='seed',
                        type=int, default=0, help='use pretrained weights')
    parser.add_argument('--tag', dest='tag',
                        type=str, default='default_tag', help='tag')
    parser.add_argument('--cfg', dest='config_file',
                        type=str, default='toy_realworld_mnist_cfg.json', help='name of the config file')
    parser.add_argument('--allowOverlap', dest='allowOverlap',
                        action='store_true', default=False, help='allow client to exchange with multiple clients at once')
    parser.add_argument('--repeat', dest='repeat',
                        type=int, default=1, help='repeat the encounter pattern')
    parser.add_argument('--upto', dest='upto',
                        type=int, default=sys.maxsize, help='number of indices on enc. data to run sim.')
    parser.add_argument('--cont', dest='cont',
                        action='store_true', default=False, help='continue from last ran swarm files')

    parsed = parser.parse_args()
    allowOverlap = parsed.allowOverlap

    if parsed.config_file == None or parsed.tag == None:
        print('Config file and the tag has to be specified. Run \'python delegation_.py -h\' for help/.')
        
    LOG_FILE_PATH = Path(LOG_FOLDER, parsed.tag + '.log')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                              "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)

    # if LOG_FILE_PATH.exists():
    #     ans = input("Simulation under the same tag already exists. Do you want to proceed? [y/N]: ")
    #     if not (ans == 'y' or ans == 'Y'):
    #         print('exit simulation.')
    #         exit()
    try:
        with open('configs/workstation_info.json', 'rb') as f:
            wsinfo_json = f.read()
        wsinfo = json.loads(wsinfo_json)
        wsinfo['workstation-name']
    except:
        print("create file \'configs/workstation_info.json\'")

    logging.basicConfig(filename=LOG_FILE_PATH, filemode='w', 
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    np.random.seed(parsed.seed)
    tf.compat.v1.set_random_seed(parsed.seed)

    # load config file
    with open(parsed.config_file, 'rb') as f:
        config_json = f.read()
    config = json.loads(config_json)

    logging.info('-----------------------<config file>-----------------------')
    for k in config:  
        logging.info(str(k + ':'))
        logging.info('    ' + str(config[k]))

    if config['dataset'] == 'mnist':
        num_classes = 10
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_mnist_dataset()
        
    elif config['dataset'] == 'cifar':
        num_classes = 10
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_cifar_dataset()
    elif config['dataset'] == 'svhn':
        num_classes = 10
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_svhn_dataset('data/svhn/')
    elif config['dataset'] == 'opportunity-uci':
        model_fn = custom_models.get_deep_conv_lstm_model
        x_train, y_train_orig, x_test, y_test_orig = get_opp_uci_dataset('data/opportunity-uci/oppChallenge_gestures.data',
                                                                         SLIDING_WINDOW_LENGTH,
                                                                         SLIDING_WINDOW_STEP)
    else:
        print("invalid dataset name")
        return

    CLIENT_NUM = config['client-num']

    enc_exp_config = config['enc-exp-config']
    hyperparams = config['hyperparams']
    hyperparams['dataset'] = config['dataset']
    hyperparams['enc-exp-config'] = enc_exp_config

    test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['hyperparams']['test-data-per-label'])

    test_swarms = []
    swarm_names = []

    OPTIMIZER = keras.optimizers.Adam
    def log_callback(message):
        log_and_upload(message, wsinfo['workstation-name'], parsed.tag, LOG_FILE_PATH)

    swarm_filename_prefix = 'swarm_'
    swarm_filename = swarm_filename_prefix + parsed.tag + '_orig.pickle'
    if not parsed.cont:
        orig_swarm = DROppCLSwarm(model_fn,
                        OPTIMIZER,
                        device.exp_device.LocalDevice,
                        CLIENT_NUM,
                        x_train,
                        y_train_orig,
                        test_data_provider,
                        0,
                        0,
                        config['local-data-size'],
                        enc_exp_config,
                        hyperparams,
                        None,
                        log_callback
                        )

    else:
        orig_swarm = None

    for k in config['strategies'].keys():
        if config['strategies'][k]:
            swarm_names.append(k)
            device_class = get_device_class(k)
            test_swarms.append(device_class)

    hists = {}
    for i in range(0, len(test_swarms)):
        start = timer()
        print("{} == running {} ".format(swarm_names[i], test_swarms[i].__name__))
        print("swarm {} of {}".format(i+1, len(test_swarms)))
        log_and_upload('starting running swarm {}'.format(i), wsinfo['workstation-name'], 
                        parsed.tag, LOG_FILE_PATH)
        if parsed.cont:
            cur_swarm = load_swarm(i, swarm_filename_prefix, parsed.tag)
        else:
            cur_swarm = get_swarm(orig_swarm, test_swarms[i], model_fn, OPTIMIZER, CLIENT_NUM, x_train, y_train_orig,
                                test_data_provider, config, enc_exp_config, hyperparams, log_callback)
        for rep in range(parsed.repeat):
            log_callback('---------- rep {} ----------'.format(rep))
            cur_swarm.run(parsed.upto, allowOverlap)
        end = timer()
        print('-------------- Elasped Time --------------')
        print(end - start)
        hists[swarm_names[i]] = copy.deepcopy(cur_swarm.hist)
        # save_swarm(cur_swarm, i, swarm_filename_prefix, parsed.tag)
        del cur_swarm

        hist_file_path = PurePath(HIST_FOLDER, 'partial_{}_'.format(i) + parsed.tag + '.pickle')
        if i > 0:
            os.remove(PurePath(HIST_FOLDER, 'partial_{}_'.format(i-1) + parsed.tag + '.pickle'))
        if i == len(test_swarms) - 1:
            hist_file_path = PurePath(HIST_FOLDER, parsed.tag + '.pickle')

        with open(hist_file_path, 'wb') as handle:
            pickle.dump(hists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # upload to S3 storage
    upload_log_path = PurePath(wsinfo['workstation-name'], 'logs/' + parsed.tag + '.log')
    client.upload_file(str(LOG_FILE_PATH), S3_BUCKET_NAME, str(upload_log_path))
    upload_hist_path = PurePath(wsinfo['workstation-name'], 'hists/' + parsed.tag + '.pickle')
    client.upload_file(str(hist_file_path), S3_BUCKET_NAME, str(upload_hist_path))

    return

    print('drawing graph...')
    processed_hists = {}
    for k in hists.keys():
        # if 'federated' in k:
        #     continue
        t, acc = get_accs_over_time(hists[k], 'clients')
        processed_hists[k] = {}
        processed_hists[k]['times'] = t
        processed_hists[k]['accs'] = acc

    for k in processed_hists.keys():
        # if 'federated' in k:
        #     continue
        plt.plot(np.array(processed_hists[k]['times']), np.array(processed_hists[k]['accs']), lw=1.2)
    plt.legend(list(processed_hists.keys()))
    if hyperparams['evaluation-metrics'] == 'f1-score-weighted':
        plt.ylabel("F1-score")
    else:
        plt.ylabel("Accuracy")
    plt.xlabel("Time")
    graph_file_path = PurePath(FIG_FOLDER, parsed.tag + '.pdf')
    plt.savefig(graph_file_path)
    plt.close()

    logging.info('Simulation completed successfully.')

    upload_graph_path = PurePath(wsinfo['workstation-name'], 'figs/' + parsed.tag + '.pdf')
    client.upload_file(str(graph_file_path), S3_BUCKET_NAME, str(upload_graph_path))

def get_accs_over_time(loaded_hist, key):
    loss_diff_at_time = []
#     print("total exchanges: {}".format(loaded_hist['total_exchanges'][-1]))
    for k in loaded_hist[key].keys():
        i = 0
        for t, h, _ in loaded_hist[key][k]:
            if t != 0:
                loss_diff_at_time.append((t, loaded_hist[key][k][i][1][1] - loaded_hist[key][k][i-1][1][1]))
            i += 1
    loss_diff_at_time.sort(key=lambda x: x[0])

    # concatenate duplicate time stamps
    ldat_nodup = []
    for lt in loss_diff_at_time:
        if len(ldat_nodup) != 0 and ldat_nodup[-1][0] == lt[0]:
            ldat_nodup[-1] = (ldat_nodup[-1][0], ldat_nodup[-1][1] + lt[1])
        else:
            ldat_nodup.append(lt)
    times = []
    loss_list = []
    times.append(0)
    # get first accuracies
    accum = []
    for c in loaded_hist[key].keys():
        accum.append(loaded_hist[key][c][0][1][1])
        
    loss_list.append(sum(accum)/len(accum))
    for i in range(1, len(ldat_nodup)):
        times.append(ldat_nodup[i][0])
        loss_list.append(loss_list[i-1] + ldat_nodup[i][1]/len(loaded_hist[key]))
        
    return times, loss_list

def log_and_upload(message, bucket, tag, log_file_path):
    logging.info(message)
    upload_log_path = PurePath(bucket, 'logs/' + tag + '.log')
    client.upload_file(str(log_file_path), S3_BUCKET_NAME, str(upload_log_path))

def get_swarm(orig_swarm, device_class, 
                model_fn, OPTIMIZER, CLIENT_NUM, x_train, y_train_orig, test_data_provider,
                config, enc_exp_config, hyperparams, log_callback):
    return DROppCLSwarm(
                        model_fn,
                        OPTIMIZER,
                        device_class,
                        CLIENT_NUM,
                        x_train,
                        y_train_orig,
                        test_data_provider,
                        0,
                        0,
                        config['local-data-size'],
                        enc_exp_config,
                        hyperparams,
                        orig_swarm,
                        log_callback
                    )

def save_swarm(swarm, num, swarm_prefix, tag):
    swarm.delete_local_objects()
    swarm_filename = swarm_prefix + str(num) + '_' + tag + '.pickle'
    try:
        with open(swarm_filename, 'wb') as handle:
            pickle.dump(swarm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('error while storing pickle')

def load_swarm(num, swarm_prefix, tag):
    swarm_filename = swarm_prefix + str(num) + '_' + tag + '.pickle'
    with open(swarm_filename, 'rb') as handle:
        sw = pickle.load(handle)
    return sw

if __name__ == '__main__':
    main()
