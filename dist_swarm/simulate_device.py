import re
import sys
from dist_swarm.db_bridge.rds_bridge import RDSCursor
from dist_swarm.db_bridge.worker_in_db import TaskInRDS, WorkerInDB
sys.path.insert(0,'..')
import os
import logging
import json
import threading
import traceback
from io import StringIO

from ovm_utils.task_runner import run_task
from grpc_components import simulate_device_pb2_grpc
from grpc_components.simulate_device_pb2 import Empty, Status
from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED, STOPPED, INITIALIZED, NOT_INITIALIZED
from dist_swarm.db_bridge.worker_in_db import WorkerInRDS
from oppcl_device import OppCLDevice

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class SimulateDeviceServicer(simulate_device_pb2_grpc.SimulateDeviceServicer):
    def __init__(self) -> None:
        super().__init__()
        # temporarily hold device state in case the next task uses this.
        # which means that this dict is erased whenever a task is ran
        self.cache = {'device_states' : {}}  
        self.current_task = []
        self.is_initialized = False

    ### gRPC methods
    def SetWorkerInfo(self, request, context):
        self.cache = {'device_states' : {}}   # clear cache
        self.worker_id = request.worker_id
        self.cursor = RDSCursor(request.rds_host, request.rds_dbname, request.rds_user,
                                request.rds_password, request.rds_table)
        self.task_cursor = RDSCursor(request.rds_host, request.rds_dbname, request.rds_user,
                                request.rds_password, request.swarm_name+'_finished_tasks')
        self.worker_db = WorkerInRDS(self.cursor, [])
        self.task_db = TaskInRDS(self.task_cursor)
        self.worker_status = STOPPED
        self.is_initialized = True
        # self.worker_in_db = WorkerInDB(request.swarm_name, request.worker_namespace, request.worker_id)
        # self.worker_status = self.worker_in_db.status
        logging.info(f"worker id set to {self.worker_id}")
        return Status(status=self.worker_status)
    
    def CheckInitialized(self, request, context):
        if self.is_initialized:
            return Status(status=INITIALIZED)
        else:
            return Status(status=NOT_INITIALIZED)

    def RunTask(self, request, context):
        config = json.load(StringIO(request.config))
        # call the function to run a single training task
        try:
            run_task_thread = threading.Thread(target=run_task, 
                                args=(self.worker_db, self.task_db, self.worker_status, 
                                    self.worker_id, config, self.cache))
            run_task_thread.start()
            self.current_task = [run_task_thread]
            return Status(status=self.worker_status)
        except Exception as e:
            logging.error(traceback.format_exc())
            return Status(status=self.worker_status)

    def CheckRunning(self, request, context):
        # checks if thread is still running
        # does not update DB
        if len(self.current_task) == 0 or \
            not self.current_task[0].is_alive():
            return Status(status=IDLE)
        return Status(status=RUNNING)

    # def ResetState(self, request, context):
    #     # kills the thread if running and reset state on DB
    #     if len(self.current_task) != 0:

    
    def ClearCache(self, request, context):
        self.device_state_cache = {}
        return Empty()
    
    def StopTask(self, request, context):
        raise NotImplementedError('')

    def GetStatus(self, request, context):
        return Status(status=self.worker_status)
    
    def SimulateOppCL(self, request, context):
        # deprecated
        # simulate a single oppcl client
        device = OppCLDevice(json.load(StringIO(request.config)))
        return device.run()
        
        
    ### helper methods
    # def run_training_task(config):
        

