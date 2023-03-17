import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import os
import boto3
import grpc
import argparse
import json
import numpy as np
import time
from decimal import Decimal
from pathlib import Path, PurePath
from pandas import read_pickle, read_csv

import data_process as dp
from get_dataset import get_dataset
import models as custom_models
from dynamo_db import DEVICE_ID, GOAL_DIST, HOSTNAME, LOCAL_DIST, MODEL_INFO, TOTAL_ENC_IDX,\
    DATA_INDICES, EVAL_HIST_LOSS, EVAL_HIST_METRIC, ENC_IDX, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, MODEL_INFO
from grpc_components.status import STOPPED
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc
from aws_settings import REGION
from get_model import get_model_fn

client = boto3.client('dynamodb', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)

class DistSwarm():
    def __init__(self, config_file, ip) -> None:
        self._config_db(config_file)
        self._config_client(ip)
        
    # deploy and run workers
    def run(self):
        # print('number of devices: {}'.format(self.config['swarm_config']['number_of_devices']))
        for device_id in range(int(self.config['swarm_config']['number_of_devices'])):
            self.config['device_config']['id'] = device_id
            config = simulate_device_pb2.Config(config=json.dumps(self.config))
            self.send_request_rep(config, device_id, 10)
            # time.sleep(1)

    def send_request_rep(self, config, device_id, failure_num):
        succeeded = False
    
        while not succeeded and failure_num > 0:
            try:
                self.send_request(config, device_id)
                failure_num -= 1
                succeeded = True
            except grpc._channel._InactiveRpcError:
                print("Inactive RPC Error")
                pass
            else:
                succeeded = True

    def send_request(self, config, device_id):
        with grpc.insecure_channel(self.loadbalancer_ip, options=(('grpc.enable_http_proxy', 0),)) as channel:
            stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
            status = stub.SimulateOppCL(config)
            # print('device {}: {}'.format(device_id, status))

    def _create_table(self, tag):
        # boto3 is the AWS SDK library for Python.
        # We can use the low-level client to make API calls to DynamoDB.

        try:
            resp = client.create_table(
                TableName=tag,
                # Declare your Primary Key in the KeySchema argument
                KeySchema=[
                    {
                        "AttributeName": DEVICE_ID,
                        "KeyType": "HASH"
                    }
                ],
                # Any attributes used in KeySchema or Indexes must be declared in AttributeDefinitions
                AttributeDefinitions=[
                    {
                        "AttributeName": DEVICE_ID,
                        "AttributeType": "N"
                    }
                ],
                # ProvisionedThroughput controls the amount of data you can read or write to DynamoDB per second.
                # You can control read and write capacity independently.
                ProvisionedThroughput={
                    "ReadCapacityUnits": 100,
                    "WriteCapacityUnits": 100
                }
                # BillingMode='PAY_PER_REQUEST'
            )
            print("Table created successfully. Syncing...")
            time.sleep(3)

        except Exception as e:
            pass
            # print("Error creating table:")
            # print(e)

        # # delete all the existing items in the db
        try:
            table = dynamodb.Table(tag)
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

    def _config_client(self, ip):
        self.loadbalancer_ip = ip

    def _config_db(self, config_file):
        # load config file
        with open(config_file, 'rb') as f:
            config_json = f.read()
        config = json.loads(config_json)
        self.config = config
        tag = config['tag']
        swarm_config = config['swarm_config']

        x_train, y_train_orig, x_test, y_test_orig = get_dataset(config['dataset'])
        num_classes = len(np.unique(y_train_orig))

        self._create_table(tag)

        enc_dataset_filename = self.config['device_config']['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)

        # store local data dist, goal dist, and trining data indices in the table
        s3 = boto3.resource('s3')
        for idnum in range(swarm_config['number_of_devices']):
            # print('init db for device {}.'.format(idnum))
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
            table = dynamodb.Table(tag)
            with table.batch_writer() as batch:
                batch.put_item(Item={DEVICE_ID: idnum, DEV_STATUS: STOPPED, TIMESTAMPS: [],
                    GOAL_DIST: convert_to_map(goal_dist),
                    LOCAL_DIST: convert_to_map(local_dist), DATA_INDICES: chosen_data_idx, TOTAL_ENC_IDX: len(enc_df.index),
                    EVAL_HIST_LOSS: [], EVAL_HIST_METRIC: [], ENC_IDX: -1, ERROR_TRACE: {}, HOSTNAME: 'N/A', MODEL_INFO: {}})
            
            # initialize and store model in S3
            model = get_model_fn(config['model'])()
            init_model_path = '.tmp/init_model.h5'
            model.save(init_model_path)
            s3.meta.client.upload_file(init_model_path, 
                                      'opfl-sim-models', 
                                       config['tag'] + '/' + 'model-' + str(idnum) + '.h5',
                                       {'Metadata': {'enc-id': '-1'}})
            del model


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
    dist_swarm = DistSwarm('../configs/dist_swarm/controller_example.json', '')