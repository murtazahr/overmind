import sys
sys.path.insert(0,'..')

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from dynamo_db import DEVICE_ID, EVAL_HIST_LOSS, EVAL_HIST_METRIC, HOSTNAME, \
            GOAL_DIST, LOCAL_DIST, DATA_INDICES, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, ENC_IDX, MODEL_INFO
# from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED

class DBController:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

    def list_tables(self):
        return [tb.name for tb in list(self.dynamodb.tables.all())]

    def get_items(self, table):
        client = boto3.client('dynamodb', region_name='us-east-2')
        paginator = client.get_paginator('scan')
        try:
            pages = paginator.paginate(TableName=table)
            res = []
            for page in pages:
                for item in page['Items']:
                    res.append(item)
            return res
        except:
            print("error while fetching items")
            return []