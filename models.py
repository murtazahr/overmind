import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Model
import tensorflow as tf
import random
import time
import numpy as np
import tensorflow as tf
# from qkeras import QDense, quantized_bits, QActivation

import models

# hyperparams for uci dataset
NUM_FILTERS = 64
FILTER_SIZE = 5
SLIDING_WINDOW_LENGTH = 24
NB_SENSOR_CHANNELS = 113
NUM_UNITS_LSTM = 128
NUM_CLASSES = 18

def get_model(dataset):
    if dataset == 'mnist':
        return models.get_2nn_mnist_model
    elif dataset == 'cifar':
        return models.get_hetero_cnn_cifar_model
    elif dataset == 'svhn':
        return models.get_hetero_cnn_cifar_model
    elif dataset == 'opportunity-uci':
        return models.get_deep_conv_lstm_model
    elif dataset == 'cifar-mobilenetv2':
        return models.get_mobilenetv2
    elif dataset == 'cifar-resnet':
        return models.get_resnet18
    elif dataset == 'femnist':
        return models.get_cnn_femnist_model

def get_2nn_mnist_model(size=-1):
    if size <= 0:
        model = Sequential()
        model.add(Flatten(input_shape=(28,28,1)))
        model.add(Dense(200, activation='relu', name='dense_0'))
        model.add(Dense(200, activation='relu', name='dense_1'))
        model.add(Dense(10, activation='softmax', name='softmax_logits'))
        return model

    return get_hetero_2nn_mnist_model(size=size)

def get_hetero_2nn_mnist_model(compressed_ver=0, size=10):
    if compressed_ver == 1:
        return get_compressed_2nn_mnist_model()
    elif compressed_ver == 2:
        return get_v2_compressed_2nn_mnist_model()
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(20 * size, activation='relu', name='dense_0'))
    model.add(Dense(20 * size, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='softmax_logits'))
    return model

def get_2nn_svhn_model(size=10):
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(int(100 - (10 - size) * 0.4), activation='relu', name='dense_0'))
    model.add(Dense(10 * size, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='softmax_logits'))
    return model

def get_mobilenetv2():
    base_model = keras.applications.MobileNetV2(input_shape=(32,32,3), weights=None, include_top=False)
    maxpool_layer = keras.layers.GlobalMaxPooling2D()
    prediction_layer = Dense(units=10, activation='softmax')
    # Layer classification head with feature detector
    model = Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    return model

# def get_Q_2nn_mnist_model(size=10):
#     bits = 4
#     integer = 0
#     symmetric = 1
#     params = '{},{},{}'.format(bits, integer, symmetric)
#     model = Sequential()
#     model.add(Flatten(input_shape=(28,28,1)))
#     model.add(QDense(5 * size, 
#                     kernel_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     bias_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     activation='quantized_relu(4,0,1)'))
#     model.add(QDense(5 * size, 
#                     kernel_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     bias_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     activation='quantized_relu(4,0,1)'))
#     model.add(QDense(10, 
#                     kernel_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     bias_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric)))
#     model.add(Activation("softmax"))
#     return model

# def get_Q_2nn_mnist_dict(size=10):
#     q_dict = {
#         "QDense": {
#             "kernel_quantizer": "quantized_bits(4,0,1)",
#             "bias_quantizer": "quantized_bits(4,0,1)"
#         }
#     }
#     return q_dict

def get_2nn_mnist_model_distill():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation=None))
    return model

def get_paramed_2nn_mnist_model(num_neurons=200):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_compressed_2nn_mnist_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_v2_compressed_2nn_mnist_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_cifar_model(size=3):
    if size == 1:
        return get_2nn_cifar_model()
    elif size == 3:
        return get_cnn_cifar_model()
    elif size == 6:
        return get_big_cnn_cifar_model()
    
    return get_big_cnn_cifar_model()

def get_2nn_cifar_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_cnn_mnist_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    return model

def get_cnn_femnist_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(62, activation='softmax'))
    return model

def get_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_bin_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def get_big_cnn_cifar_model():
    model = Sequential()
    RAND_MAX = 9999999
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3), name='layer_{}'.format(random.randint(RAND_MAX))))
    model.add(Activation('relu'), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Conv2D(32, (3, 3), name='layer_{}'.format(random.randint(RAND_MAX))))
    model.add(Activation('relu'), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(MaxPooling2D(pool_size=(2, 2), name='layer_{}'.format(random.randint(RAND_MAX))))
    model.add(Dropout(0.25), name='layer_{}'.format(random.randint(RAND_MAX)))

    model.add(Conv2D(64, (3, 3), padding='same', name='layer_{}'.format(random.randint(RAND_MAX))))
    model.add(Activation('relu'), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Conv2D(64, (3, 3), name='layer_{}'.format(random.randint(RAND_MAX))))
    model.add(Activation('relu'), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(MaxPooling2D(pool_size=(2, 2), name='layer_{}'.format(random.randint(RAND_MAX))))
    model.add(Dropout(0.25), name='layer_{}'.format(random.randint(RAND_MAX)))

    model.add(Flatten(), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Dense(512), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Activation('relu'), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Dropout(0.5), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Dense(10), name='layer_{}'.format(random.randint(RAND_MAX)))
    model.add(Activation('softmax'), name='layer_{}'.format(random.randint(RAND_MAX)))
    return model

def get_all_cnn_c_model(size=10):
    model = Sequential()
    dr = {5: 0.5, 6: 0.625, 7: 0.750, 8: 0.875, 9: 0.875, 10: 1}
    if size not in dr:
        raise ValueError('CNN model does not support model size {}'.format(size))
    small_conv_fs = 96 * dr[size]
    big_conv_fs = 192 * dr[size]

    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(small_conv_fs, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(small_conv_fs, (3, 3), padding='same', strides = (2,2)))

    model.add(Conv2D(big_conv_fs, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(big_conv_fs, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(big_conv_fs, (3, 3),padding='same', strides = (2,2)))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    return model

def get_rand_layer_name():
    RAND_MAX = 999999
    cur_t = time.time()
    randnum = random.randint(0,RAND_MAX) + cur_t
    return 'layer_{}'.format(randnum)

def get_hetero_cnn_cifar_model(size = 3):
    model = Sequential()
    RAND_MAX = 999999
    if size == 3:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Conv2D(32, (3, 3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(MaxPooling2D(pool_size=(2, 2), name=get_rand_layer_name()))

        model.add(Conv2D(64, (3, 3), padding='same', name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Conv2D(64, (3, 3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(MaxPooling2D(pool_size=(2, 2), name=get_rand_layer_name()))

        model.add(Flatten( name=get_rand_layer_name()))
        model.add(Dense(512, name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Dense(10, name=get_rand_layer_name()))
        model.add(Activation('softmax', name=get_rand_layer_name()))
        
    elif size == 2:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Conv2D(32, (3, 3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(MaxPooling2D(pool_size=(2, 2), name=get_rand_layer_name()))

        model.add(Conv2D(48, (3, 3), padding='same', name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Conv2D(48, (3, 3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(MaxPooling2D(pool_size=(2, 2), name=get_rand_layer_name()))

        model.add(Flatten( name=get_rand_layer_name()))
        model.add(Dense(512, name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Dense(10, name=get_rand_layer_name()))
        model.add(Activation('softmax', name=get_rand_layer_name()))
        
    elif size == 1:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Conv2D(32, (3, 3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(MaxPooling2D(pool_size=(2, 2), name=get_rand_layer_name()))

        model.add(Conv2D(32, (3, 3), padding='same', name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Conv2D(32, (3, 3), name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(MaxPooling2D(pool_size=(2, 2), name=get_rand_layer_name()))

        model.add(Flatten( name=get_rand_layer_name()))
        model.add(Dense(256, name=get_rand_layer_name()))
        model.add(Activation('relu', name=get_rand_layer_name()))
        model.add(Dense(10, name=get_rand_layer_name()))
        model.add(Activation('softmax', name=get_rand_layer_name()))
    else:
        raise ValueError("size {} is not supported cnn model size".format(size))
    
    return model

def get_deep_hetero_cnn_cifar_model(size = 3):
    model = Sequential()
    if size >= 3:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
    elif size == 2:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
    elif size == 1:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
    else:
        raise ValueError("size {} is not supported cnn model size".format(size))
    
    return model

def get_big_bin_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def get_big_quad_cnn_cifar_model(compressed_ver=0):
    if compressed_ver == 1:
        return get_compressed_big_quad_cnn_cifar_model()
    elif compressed_ver == 2:
        return get_v2_compressed_big_quad_cnn_cifar_model()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def get_compressed_big_quad_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def get_v2_compressed_big_quad_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def get_better_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def get_deep_conv_lstm_model(num_filters=NUM_FILTERS,
                             filter_size=FILTER_SIZE,
                             sliding_window_length=SLIDING_WINDOW_LENGTH,
                             nb_sensor_channels=NB_SENSOR_CHANNELS,
                             num_units_lstm=NUM_UNITS_LSTM,
                             num_classes=NUM_CLASSES):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu', name='conv1',
                                  input_shape=(sliding_window_length, nb_sensor_channels, 1)))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu', name='conv2'))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu', name='conv3'))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu', name='conv4'))
    shape = model.layers[-1].output_shape
    model.add(keras.layers.Reshape((shape[1], shape[3] * shape[2])))
    model.add(keras.layers.LSTM(num_units_lstm, activation='tanh', return_sequences=True, name='lstm1')) # [batch, timesteps, features]
    model.add(keras.layers.Dropout(0.5, seed=123, name='dr1'))
    model.add(keras.layers.LSTM(num_units_lstm, activation='tanh', name='lstm2'))
    model.add(keras.layers.Dropout(0.5, seed=124, name='dr2'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

def get_resnet18():
    model = ResNet18(10)
    model.build(input_shape = (None,32,32,3))
    return model