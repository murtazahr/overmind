import tensorflow.keras as keras

def get_optimizer(name):
    pname = name.lower()
    if pname == 'sgd':
        return keras.optimizers.SGD
    elif pname == 'adam':
        return keras.optimizers.Adam
    elif pname == 'adadelta':
        return keras.optimizers.Adadelta
