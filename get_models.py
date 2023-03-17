import numpy as np

def get_init_model_with_accuracy(model_fn, args, compile_config, x_test, y_test, acc, error):
    TRY_LIMIT = 40
    print('initializing models...')
    for i in range(TRY_LIMIT):
        m = model_fn(**args)
        m.compile(**compile_config)
        hist = m.evaluate(x_test, y_test)
        if np.abs(hist[1] - acc) <= error:
            print('got init model for [{}] in {} trials.'.format(args, i))
            return m.get_weights()
    raise RuntimeError('Failed to get initial weights')