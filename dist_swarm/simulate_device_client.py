import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import logging
import json
import grpc
import argparse

from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc

def run():
    # parse arguments
    parser = argparse.ArgumentParser(description='set params for simulation')
    parser.add_argument('--target', dest='target',
                        type=str, default='localhost:50051', help='target ip addr and port')

    with open('../configs/dist_swarm/droppcl_svhn.json', 'rb') as f:
        config_json = f.read()
    config_json = json.loads(config_json)

    parsed = parser.parse_args()

    with grpc.insecure_channel(parsed.target) as channel:
        stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
        config = simulate_device_pb2.Config(config=json.dumps(config_json))
        status = stub.SimulateOppCL(config)
        print(status)

if __name__ == '__main__':
    logging.basicConfig()
    run()