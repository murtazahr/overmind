import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import grpc
import logging
from concurrent import futures
from simulate_device import SimulateDeviceServicer
from grpc_components import simulate_device_pb2_grpc

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulate_device_pb2_grpc.add_SimulateDeviceServicer_to_server(
        SimulateDeviceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(filename="simulate_device.log", 
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        level=logging.INFO)
    serve()