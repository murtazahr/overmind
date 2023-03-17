from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from utils.har.sliding_window import sliding_window
from scipy.io import loadmat
import _pickle as cp
import boto3
import json
from pathlib import Path

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

def get_dataset(name, client_num=0):
    """
    returns x_train, y_train_orig, x_test, y_test_orig 
    """
    if name == 'mnist':
        return get_mnist_dataset()       
    elif name.split('-')[0] == 'cifar':
        return get_cifar_dataset()
    elif name == 'svhn':
        return get_svhn_dataset('data/svhn/')
    elif name == 'opportunity-uci':
        return get_opp_uci_dataset('data/opportunity-uci/oppChallenge_gestures.data',
                                            SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    elif name == 'femnist':
        return get_femnist_dataset(client_num)
    else:
        raise ValueError('No such dataset: {}'.format(name))

def get_femnist_dataset(client_num):
    path = '../leaf/data/femnist/data/all_data/'
    file_num = (int) (client_num / 100)
    filename = f'all_data_{file_num}.json'
    with open(path + filename, 'rb') as f:
        data_json = f.read()
    data = json.loads(data_json)
    idx = client_num - file_num * 100
    user_name = data['users'][idx]
    data_size = data['num_samples'][idx]
    x = data['user_data'][user_name]['x']
    y = data['user_data'][user_name]['y']
    x_train = (np.array(x)).reshape(data_size, 28, 28, 1)
    y_train = (np.array(y))

    return x_train, y_train, 0, 0


def get_mnist_dataset():
    # import dataset
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    return x_train, y_train_orig, x_test, y_test_orig

def get_cifar_dataset():
    img_rows, img_cols = 32, 32
    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = tf.keras.datasets.cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train_orig = y_train_orig.reshape(-1)
    y_test_orig = y_test_orig.reshape(-1)

    return x_train, y_train_orig, x_test, y_test_orig

def get_cifar100_dataset(label_mode='fine'):
    img_rows, img_cols = 32, 32
    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = tf.keras.datasets.cifar100.load_data(label_mode=label_mode)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train_orig = y_train_orig.reshape(-1)
    y_test_orig = y_test_orig.reshape(-1)

    return x_train, y_train_orig, x_test, y_test_orig

def get_svhn_dataset(path):
    filepath = Path(path + 'train_32x32.mat')
    if not filepath.is_file():
        download_svhn()
        print('downloading dataset...')

    train_raw = loadmat(path + 'train_32x32.mat')
    test_raw = loadmat(path + 'test_32x32.mat')
    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])

    train_labels = train_raw['y']
    test_labels = test_raw['y']

    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)
    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')
    train_labels = (train_labels.astype('int64') - 1).reshape(-1)
    test_labels = (test_labels.astype('int64') - 1).reshape(-1)
    train_images /= 255.0
    test_images /= 255.0

    return train_images, train_labels, test_images, test_labels

def get_opp_uci_dataset(filename, sliding_window_length, sliding_window_step):
    # from https://github.com/STRCWearlab/DeepConvLSTM

    filepath = Path(filename)
    if not filepath.is_file():
        download_uci_opportunity()
        print('downloading dataset...')

    with open(filename, 'rb') as f:
        data = cp.load(f)

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    X_train, Y_train = opp_sliding_window(X_train, y_train, sliding_window_length, sliding_window_step)
    X_test, Y_test = opp_sliding_window(X_test, y_test, sliding_window_length, sliding_window_step) 

    return  np.expand_dims(X_train, axis=3), Y_train,  np.expand_dims(X_test, axis=3), Y_test

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def download_from_s3(bucket_name, data_path, file_name):
    client = boto3.client('s3')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    client.download_file(bucket_name, file_name, data_path + '/' + file_name)


def download_uci_opportunity():
    download_from_s3('opfl-sim-models', 'data/opportunity-uci', 'oppChallenge_gestures.data')

def download_svhn():
    download_from_s3('opfl-sim-models', 'data/svhn', 'train_32x32.mat')
    download_from_s3('opfl-sim-models', 'data/svhn', 'test_32x32.mat')