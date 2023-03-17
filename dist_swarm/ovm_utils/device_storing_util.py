# functions for storing device on AWS
# Device is divided into four parts: 
#       device state (encounter idx, goal/label distribution)
#       device file (pickled device object. attributes of the device. make sure to sync with dev state)
#                   (model and dataset is deleted when saved and retrieved later)
#       device model (just the model parameters)
#       device dataset (dataset indices)
#
import boto3
import pickle
import tensorflow.keras as keras
from pathlib import Path
import logging

from device import base_device
from dist_swarm.aws_settings import REGION
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from dist_swarm.db_bridge.model_in_db import BUCKET_NAME

dynamodb = boto3.resource('dynamodb', region_name=REGION)

def get_device_model_object_name(tag, id):
    return tag + '/model-' + str(id) + '.h5'

def get_device_model_weight_name(tag, id):
    return tag + '/model-' + str(id) + '.pickle'

def get_device_file_name(tag, id):
    return tag + '/device-' + str(id) + '.pickle'

def get_device_resource_object_name(tag, name, id):
    return tag + '/obj-' + name + '-' + str(id) + '.pickle'

def save_device_as_pickle(device: base_device.Device, path, swarm_name, id, enc_idx, overwrite=True, bucket='opfl-sim-models'):
    # store pickled device to S3
    s3 = boto3.resource('s3')
    tmp_device_path = path + f"/device_{id}.pickle"
    if overwrite or not Path(tmp_device_path).exists():
        with open(tmp_device_path, 'wb') as handle:
            pickle.dump(device, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    s3.meta.client.upload_file(tmp_device_path, 
                                bucket, 
                                swarm_name + '/' + 'device-' + str(id) + '.pickle',
                                {'Metadata': {'enc-id': str(enc_idx)}})

    Path(tmp_device_path).unlink()

def load_device_as_pickle(tag, id, bucket='opfl-sim-models'):
    s3 = boto3.resource('s3')
    tmp_device_path = ".tmp/device.pickle"
    s3.Bucket(bucket).download_file(get_device_file_name(tag, id), tmp_device_path)
    with open(tmp_device_path, 'rb') as handle:
        dev = pickle.load(handle)
    Path(tmp_device_path).unlink()
    return dev

def save_device_model(device: base_device.Device, path, swarm_name, id, enc_idx, overwrite=True, bucket='opfl-sim-models'):
    s3 = boto3.resource('s3')
    model = device._model_fn()
    model.set_weights(device._weights)
    # init_model_path = path + f'/init_model_{id}.h5'
    init_weight_path = path + f'/init_weights_{id}.pickle'
    if overwrite or not Path(init_weight_path).exists():
        with open(init_weight_path, 'wb') as handle:
            pickle.dump(device._weights, handle)
    s3.meta.client.upload_file(init_weight_path, 
                                bucket, 
                                get_device_model_weight_name(swarm_name, id),
                                {'Metadata': {'enc-id': str(enc_idx)}})
    Path(init_weight_path).unlink()

def load_device_model(device: base_device.Device, tag, id, bucket='opfl-sim-models'):
    s3 = boto3.resource('s3')
    tmp_model_path = '.tmp/loaded_model.h5'
    try:
        s3.Bucket(bucket).download_file(get_device_model_object_name(tag, id),
                        tmp_model_path)
        model = keras.models.load_model(tmp_model_path, compile=False)
        device._weights = model.get_weights()
        Path(tmp_model_path).unlink()
    except:
        tmp_weight_path = '.tmp/loaded_weights.pickle'
        s3.Bucket(bucket).download_file(get_device_model_weight_name(tag, id),
                        tmp_weight_path)
        with open(tmp_weight_path, 'rb') as handle:
            device._weights = pickle.load(handle)
        Path(tmp_weight_path).unlink()
        
def save_device_dataset(device: base_device.Device, path, swarm_name, id, enc_idx, overwrite, bucket='opfl-sim-models'):
    save_data_object(device._x_train, "x_train", path, swarm_name, id, enc_idx, overwrite, bucket)
    save_data_object(device._y_train_orig, "y_train_orig", path, swarm_name, id, enc_idx, overwrite, bucket)
    save_data_object(device._y_train, "y_train", path, swarm_name, id, enc_idx, overwrite, bucket)
    save_data_object(device.test_data_provider, "test_data_provider", path, swarm_name, id, enc_idx, overwrite, bucket)

def save_data_object(obj, name, path, swarm_name, id, enc_idx, overwrite=True, bucket='opfl-sim-models'):
    s3 = boto3.resource('s3')
    tmp_local_file_path = path + f"/dataset_{name}_{id}.pickle"
    if overwrite or not Path(tmp_local_file_path).exists():
        with open(tmp_local_file_path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    s3.meta.client.upload_file(tmp_local_file_path, 
                                bucket, 
                                get_device_resource_object_name(swarm_name, name, id),
                                {'Metadata': {'enc-id': str(enc_idx)}})
    Path(tmp_local_file_path).unlink()

def load_and_set_attr(device, attr_name, obj_name, tag, id, bucket='opfl-sim-models'):
    s3 = boto3.resource('s3')
    tmp_local_file_path = ".tmp/dataset.pickle"
    s3.Bucket(bucket).download_file(get_device_resource_object_name(tag, obj_name, id),
                     tmp_local_file_path)
    with open(tmp_local_file_path, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    setattr(device, attr_name, loaded_obj)
    Path(tmp_local_file_path).unlink()

def load_data_objects(device, tag, id, bucket='opfl-sim-models'):
    s3 = boto3.resource('s3')
    load_and_set_attr(device, '_x_train', 'x_train', tag, id, bucket)
    load_and_set_attr(device, '_y_train_orig', 'y_train_orig', tag, id, bucket)
    load_and_set_attr(device, '_y_train', 'y_train', tag, id, bucket)
    load_and_set_attr(device, 'test_data_provider', 'test_data_provider', tag, id, bucket)
    device.set_local_data(device._x_train, device._y_train_orig)

def save_device(device: base_device.Device, swarm_name: str, swarm_init_group: str, enc_idx: int, overwrite=True, bucket='opfl-sim-models'):
    id = device._id_num
    # save device states
    # device_state = DeviceInDB(tag, id)
    # device_state.update_loss_and_metric_in_bulk(device.hist_loss, device.hist_metric, enc_idx)
    # device_state.update_timestamps_in_bulk(device.timestamps)
    # make storage for temporary files
    path = f'.tmp/{swarm_init_group}'
    Path(path).mkdir(parents=True, exist_ok=True)

    # save model
    save_device_model(device, path, swarm_name, id, enc_idx, overwrite, bucket)

    # save data
    save_device_dataset(device, path, swarm_name, id, enc_idx, overwrite, bucket)

    # strip model and data from the device and save device as pickle
    device._weights = ""
    device._x_train = ""
    device._y_train_orig = ""
    device._y_train = ""
    device.test_data_provider = ""

    # save device
    save_device_as_pickle(device, path, swarm_name, id, enc_idx, overwrite, bucket)

def load_device(tag, device_id, load_model=True, load_dataset=True, bucket='opfl-sim-models'):
    Path(".tmp").mkdir(parents=True, exist_ok=True)

    # load pickled device class
    device = load_device_as_pickle(tag, device_id, bucket)

    if load_model:
        load_device_model(device, tag, device_id, bucket)
    if load_dataset:
        load_data_objects(device, tag, device_id, bucket)

    return device
    
