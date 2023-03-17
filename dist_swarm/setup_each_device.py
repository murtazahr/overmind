def setup_each_device(config):
    strategy = config['device_config']['device_strategy'] 
    hetero_strategies = ['c_DROppCL', 'c_DROppCL Auto', 'c_only dropout',
                         'c_only quant', 'no dropout nor Q.']
    
    if strategy in hetero_strategies:
        id = config['device_config']['id']
        hyperparams = config['device_config']['train_config']
        model_sizes = hyperparams['enc-exp-config']['client-sizes']
        device_powers = hyperparams['enc-exp-config']['device-powers']

        hyperparams['model_size']