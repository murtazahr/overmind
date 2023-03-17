import models

def get_model_fn(name):
    if name == 'quantized-2nn':
        return models.get_Q_2nn_mnist_model
    elif name == '2nn':
        return models.get_hetero_2nn_mnist_model
    elif name.lower() == 'mobilenetv2':
        return models.get_mobilenetv2