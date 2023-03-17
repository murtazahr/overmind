# task runner function for simulate_device class
from decimal import Decimal
from inspect import trace
from time import gmtime, strftime
import traceback
import sys
import copy
import logging
import time
import device
from grpc_components.status import RUNNING, STOPPED

sys.path.insert(0,'..')

from ovm_utils.device_storing_util import load_device, save_device
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from dist_swarm.db_bridge.worker_in_db import WorkerInRDS
from dynamo_db import TASK_END, TASK_FAILED, TASK_START, TASK_REALTIME_TIMEOUT

# @TODO implement timeout here
def run_task(worker_db, task_db, worker_status, worker_id, task_config, device_state_cache):
    swarm_name = task_config['swarm_name']
    worker_namespace = task_config['worker_namespace']
    if worker_status != STOPPED:
        logging.info(f"RPC call for a new task {task_config['task_id']} is made while worker is busy")
        # worker_in_db.update_finished_task(task_config['task_id'], False)
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_FAILED, "task": task_config, "error_msg": traceback.format_exc()}
        return new_history, {}
    worker_db.update_status(worker_id, RUNNING)
    # worker_in_db.append_history(strftime("%Y-%m-%d %H:%M:%S", gmtime()), TASK_START, task_config)
    logging.info(f"Running task {task_config['task_id']} in worker {worker_id}")
    logging.info(f"cached is {device_state_cache['device_states'].keys()}")
    task_start_time = time.time()
    realtime_timed_out = False
    sim_timestamp = 0
    hist = [0 for _ in range(3)]
    try:
        # change the state of current worker
        task_id = task_config['task_id']
        learner_id = task_config['learner']
        neighbor_ids = task_config['neighbors']
        func_list = task_config['func_list']
        device_load_config = task_config['load_config']
        timeout = task_config['timeout']
        worker_namespace = task_config['worker_namespace']
        real_time_mode = task_config['real_time_mode']  # string: 'True' or 'False'
        real_time_timeout = float(task_config['real_time_timeout'])
        end = Decimal(task_config['end'])
        start = Decimal(task_config['start'])
        comm_time = Decimal(task_config['communication_time'])
        comp_time = Decimal(task_config['computation_time'])
        if 'bucket' in task_config:
            bucket = task_config['bucket']
        else:
            bucket = 'opfl-sim-models'
        orig_enc_idx = task_config['orig_enc_idx']
        measured_time = 0

        # load device
        learner_load_config = {}
        if str(learner_id) in device_load_config:
            learner_load_config = device_load_config[str(learner_id)]
        # if device is already in cache, use that
        # @TODO do sanity check. Cached device is not up-to-date? reused in different swarm run?
        if task_config['learning_scenario'] == 'oppcl' and str(learner_id) in device_state_cache['device_states']:
            learner = device_state_cache['device_states'][str(learner_id)]
            logging.info(f"device state {learner_id} loaded from cache")
        else:
            learner = load_device(swarm_name, learner_id, **learner_load_config, bucket)

        device_in_db = DeviceInDB(swarm_name, learner_id)
        neighbors = []
        for ngbr in neighbor_ids:
            load_config = {}
            if str(ngbr) in device_load_config:
                load_config = device_load_config[str(ngbr)]
            neighbors.append(load_device(swarm_name, ngbr, **load_config, bucket))
        
        # save to cache before saving to DB
        # because save_device deletes model and data from the device
        device_state_cache['device_states'] = {}
        device_state_cache['device_states'][str(learner_id)] = copy.deepcopy(learner)
        
        # invoke function 
        for i in range(len(func_list)):
            # computations that are measured
            if func_list[i]["func_name"][0] != '!':
                func_name = func_list[i]["func_name"]
                is_measured = True
                if func_name[0] == '#':
                    func_name = func_name[1:]
                    is_measured = False

                func = getattr(learner, func_name)
                
                # @TODO handle multiple neighbors and multiple function calls before eval()
                compute_start = time.time()
                for nbgr in neighbors:
                    print(f"{type(nbgr)} and {func_list[i]}")
                    func(nbgr, **func_list[i]["params"])
                if is_measured:
                    measured_time += time.time() - compute_start
            elif func_list[i]["func_name"] == '!evaluate' and measured_time <= timeout:
                hist = learner.eval()
                if real_time_mode == 'True' and real_time_timeout < measured_time:
                    realtime_timed_out = True

                if real_time_mode == 'False':
                    sim_timestamp = start + comp_time + comm_time
                else:
                    sim_timestamp = start + Decimal(measured_time) + comm_time
                

        # save device (s)
        if not realtime_timed_out:
            device_state_cache['device_states'][str(learner_id)] = copy.deepcopy(learner)
            save_device(learner, swarm_name, "local", task_id, True, bucket)
        # @TODO save neighbor device states if instructed
        
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_REALTIME_TIMEOUT 
                       if real_time_mode and real_time_timeout < measured_time else TASK_END, "task": task_config}
    except:
        logging.error(f'{traceback.format_exc()}')
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_FAILED, "task": task_config, "error_msg": traceback.format_exc()}
        # worker_in_db.append_history(**new_history)
        worker_db.update_status(worker_id, STOPPED)
        return new_history

    total_time = time.time() - task_start_time
    logging.info(f"-- Task {task_id} successfully finished --")
    try:
        worker_db.update_status(worker_id, STOPPED)
        # worker_in_db.append_history(**new_history)
        logging.info(f"updated status")
        task_db.insert_newly_finished_task(task_id, realtime_timed_out, measured_time, total_time,
                                           learner_id, "\'" + str(neighbor_ids) + "\'", sim_timestamp, time.time(),
                                           hist[0], hist[1], orig_enc_idx, worker_id)
        # worker_in_db.update_finished_task(task_id, True, realtime_timed_out, Decimal(measured_time))
        logging.info(f"inserted finished task")
    except:
        logging.error(f"Task {task_id} returned an error while updating status: {traceback.format_exc()}")

    return new_history
