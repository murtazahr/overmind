"""
resides in user side
"""

import json
import os
import argparse
from pathlib import Path, PurePath

class DistSwarmController():
    def __init__(self) -> None:
        self.swarm = None
        self.loadbalancer_ip = 'a00ffd26030dd49cba89f364b43d0321-154579848.us-east-2.elb.amazonaws.com:80'

    ### config methods
    def create_config(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-o', dest='out_file',
                        type=str, default=None, help='output config filename')
        parsed = parser.parse_args(args.split())
        if parsed.out_file == None:
            print('use -o option to name the output config')
            return

        # load config file
        path = PurePath(os.path.dirname(__file__) + 'configs/dist_swarm/default_config.json')
        with open(path, 'rb') as f:
            config_json = f.read()
        config = json.loads(config_json)

        out_path = PurePath(os.path.dirname(__file__) + 'configs/dist_swarm/' + parsed.out_file + '.json')
        with open(out_path, 'wb') as f:
            f.write(config)

    def edit_config(self, args):
        path = PurePath(args[0])
        with open(path, 'rb') as f:
            config_json = f.read()
        config = json.loads(config_json)
        raise NotImplementedError()

    def remove_config(self, args):
        raise NotImplementedError()

    ### k8s setup methods
    def set_loadbalancer_ip(self, ip):
        # set the external ip of loadbalancer
        self.loadbalancer_ip = ip

    ### swarm methods
    def create_swarm(self, args):

        # initialize channel
        with grpc.insecure_channel(self.loadbalancer_ip) as channel:
            self.stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
            

    def remove_swarm(self, args):
        raise NotImplementedError()

    def _config_to_grpc_msg(self, config_json):
        return simulate_device_pb2.Config(config=json.dumps(config_json))