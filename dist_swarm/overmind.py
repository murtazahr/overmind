# overmind controller
# reads encounter data, creates tasks and build dependency graph
from sqlite3 import Date
import sys
import traceback
import grpc
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import logging

from dist_swarm.db_bridge.worker_in_db import TaskInRDS, WorkerInDB, WorkerInRDS
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc
from grpc_components.status import INITIALIZED, RUNNING, STOPPED
import json
import os
import threading
import time
from time import sleep
import pickle
import datetime
import typing
from pathlib import PurePath, Path
import pandas as pd
import numpy as np
from pandas import read_pickle, read_csv
import boto3
from dynamo_db import IS_FINISHED, IS_PROCESSED, IS_TIMED_OUT, TASK_ID, TIME
from boto3.dynamodb.conditions import Key

from ovm_swarm_initializer import OVMSwarmInitializer
from get_device import get_device_class

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

MAX_ITERATIONS = 10_000_000

class Task():
    def __init__(self, swarm_name, task_id, start, end, learner_id, neighbor_id_list,
                 timeout=2**16, real_time_mode=False, communication_time=0, computation_time=0, 
                 dependant_on_mutable=False, orig_enc_idx=0, learning_config='oppcl', bucket='opfl-sim-models'):
        self.swarm_name = swarm_name
        self.worker_namespace = "deprecated"
        self.task_id = task_id
        self.start = start
        self.end = end
        self.learner_id = learner_id
        self.neighbor_id_list = neighbor_id_list
        self.timeout = timeout
        self.real_time_mode = real_time_mode
        self.computation_time = computation_time
        self.communication_time = communication_time
        self.learning_config = learning_config
        self.orig_enc_idx = orig_enc_idx
        self.bucket = bucket

        self.real_time_timeout = self.end - self.start - communication_time
        self.dependant_on_mutable = dependant_on_mutable

        self.skip = False  # skip the processing of this task
        self.func_list = []

        self.set_learner_load_config(True, False)
        self.set_neighbor_load_config(False, True)
        self.set_load_config()

    def update_real_time_timeout(self, communication_time):
        self.real_time_timeout = self.end - self.start - communication_time

    def determine_skip(self):
        if self.start >= self.end:
            self.skip = True

    def add_func(self, func_name, params):
        self.func_list.append({
            "func_name": func_name,
            "params": params,
        })

    def add_eval(self):
        self.func_list.append({"func_name": "!evaluate"})
    
    def get_config(self):
        return {
            "swarm_name": self.swarm_name,
            "worker_namespace": self.worker_namespace,
            "task_id": self.task_id,
            "start": str(self.start),
            "end": str(self.end),
            "communication_time": str(self.communication_time),
            "computation_time": str(self.computation_time),
            "learner": self.learner_id,
            "neighbors": self.neighbor_id_list,
            "load_config": self.load_config,
            "func_list": self.func_list,
            "timeout": self.timeout,  # overmind controller assumes that server is dead when timeout is elasped
            "real_time_mode": str(self.real_time_mode),
            "real_time_timeout": str(self.real_time_timeout),
            "learning_scenario": self.learning_config,
            "orig_enc_idx": str(self.orig_enc_idx),
            "bucket": str(self.bucket)
        }
        
    def set_learner_load_config(self, learner_load_model: bool, learner_load_dataset: bool):
        self.learner_load_model = learner_load_model
        self.learner_load_dataset = learner_load_dataset
    
    def set_neighbor_load_config(self, neighbor_load_model: bool, neighbor_load_dataset: bool):
        self.neighbor_load_model = neighbor_load_model
        self.neighbor_load_dataset = neighbor_load_dataset

    def get_new_load_config(self, load_model: bool = False, load_dataset: bool=True):
        return {"load_model": load_model, "load_dataset": load_dataset}

    def set_load_config(self):
        self.load_config = {str(self.learner_id): self.get_new_load_config(self.learner_load_model, self.learner_load_model)}
        for nbgr in self.neighbor_id_list:
            self.load_config[str(nbgr)] = self.get_new_load_config(self.neighbor_load_model, self.neighbor_load_dataset)

    def reset_start_time(self, last_avail):
        self.start = max(self.start, last_avail[self.learner_id])
        if self.dependant_on_mutable:
            for nid in self.neighbor_id_list:
                self.start = max(self.start, last_avail[nid])
        self.real_time_timeout = self.end - self.start - self.communication_time
    
    def is_skipped(self, process_time=0):
        return self.end - self.start < process_time
        

class Overmind():
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

    def create_swarm(self, config_file, skip_init_tables=False, init_worker=True):
        with open(config_file, 'rb') as f:
            config_json = f.read()
        self.config = json.loads(config_json)
        self.device_config = self.config["device_config"]
        self.swarm_name = self.config['tag']
        if "worker_ips" in self.config:
            self.worker_nodes : typing.List[str] = self.config["worker_ips"]
        else:
            with open(self.config["cluster_config"], 'rb') as f:
                cluster_config = f.read()
            cluster_config = json.loads(cluster_config)
            self.worker_nodes = cluster_config["worker_ips"]
            self.config["worker_ips"] = self.worker_nodes
        self.number_of_devices = self.config["swarm_config"]["number_of_devices"] + (len(self.device_config["device_groups"]) if "device_groups" in self.device_config else 0)
        self.learning_scenario = self.config["learning_scenario"]
        self.worker_namespace = "deprecated"

        if 'bucket' in self.config:
            self.bucket = self.config['bucket']
        else:
            self.bucket = 'opfl-sim-models'

        self.log_path = f'ovm_logs/{self.swarm_name}/{datetime.datetime.now()}'
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=self.log_path + '/overmind.log', 
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',level=logging.INFO)

        self.initializer = OVMSwarmInitializer()
        self.initializer.initialize(self.config, not skip_init_tables, init_worker)
        self.worker_db = WorkerInRDS(self.initializer.rds_cursor, self.worker_nodes)

        self.worker_ip_to_id = self.initializer.worker_ip_to_id
        self.worker_id_to_ip = {v: k for k, v in self.worker_ip_to_id.items()}

        self.tasks_db = TaskInRDS(self.initializer.rds_tasks_cursor)

    def load_and_run_swarm(self, device_state_log_file, task_table_log_file, etc_log_file, rt_mode=False):
        CHECKPOINT_INTERVAL = 1000
        polling_interval = 5
        with open(etc_log_file, 'rb') as handle:
            etc_log = pickle.load(handle)
        swarm_state = etc_log['run_swarm']
        self.config = etc_log['config']
        self.tasks = swarm_state['tasks']
        self.task_queue = swarm_state['task_queue']
        self.processed_tasks = swarm_state['processed_tasks']
        self.indegrees = swarm_state['indegrees']
        self.dep_graph = swarm_state['dep_graph']
        self.deployed_tasks = swarm_state['deployed_tasks']
        self.cached_devices_to_worker_nodes = {}  # (cached_device_id, worker node)
        self.cached_worker_nodes_to_devices = {}
        self.allocated_tasks = dict.fromkeys(range(len(self.worker_nodes)), 0)
        self.last_avail = dict.fromkeys(range(self.number_of_devices), 0)
        self.cur_worker_to_task_id = {}
        self.next_checkpoint = len(self.processed_tasks) + CHECKPOINT_INTERVAL
        self.successful_tasks = 0
        self.timed_out_tasks = 0

        if len(self.task_queue) == 0:
            self.task_queue = [t for t in self.deployed_tasks]

        run_swarm_start_time = time.time() - float(etc_log['elasped_time'])

        # TODO make separate routine for this duplicated code
        # recover tables from log files
        # with open(task_table_log_file, 'rb') as handle:
        #     records = pickle.load(handle)
        # columns = [
        #     'serial_id',
        #     'task_id',
        #     'is_processed',
        #     'is_finished',
        #     'is_timed_out',
        #     'sim_time',
        #     'wc_time',
        #     'learner',
        #     'neighbor',
        #     'sim_timestamp',
        #     'wc_timestamp',
        #     'loss',
        #     'metric',
        #     'enc_idx',
        #     'undefined'
        # ]
        # dtypes = np.dtype([
        #         (columns[0], int),
        #         (columns[1], int),
        #         (columns[2], bool),
        #         (columns[3], bool),
        #         (columns[4], bool),
        #         (columns[5], str),
        #         (columns[6], str),
        #         (columns[7], str),
        #         (columns[8], str),
        #         (columns[9], str),
        #         (columns[10], str),
        #         (columns[11], str),
        #         (columns[12], str),
        #         (columns[13], str),
        #         (columns[14], str),
        #         ])
        # data = np.empty(0, dtype=dtypes)
        # df = pd.DataFrame(data = data)
        # logging.info('constructing task table')
        # for r in records:
        #     row_dict = {}
        #     for i in range(len(columns)):
        #         row_dict[columns[i]] = r[i]
        #     df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = row_dict   
        
        # df.to_csv('tmp.csv', sep=',', index_label=False, index=False, header=False)
        # self.tasks_db.recover_from_file('tmp.csv')
        # logging.info('task table is recovered from the checkpoint')

        iterations = 0
        blocked_time = 0
        while iterations < MAX_ITERATIONS:
            if not Path(self.log_path).exists():  # exit simulation when log file is deleted
                return

            iterations += 1

            finished_not_processed_tasks = self.tasks_db.get_not_processed_finished_tasks()
            # print(f"not fin: {finished_not_processed_tasks}")
            # checkpointing
            if len(self.processed_tasks) >= self.next_checkpoint:
                self.save_checkpoint({'elasped_time': f'{time.time() - run_swarm_start_time}'}, len(self.processed_tasks))
                self.next_checkpoint += CHECKPOINT_INTERVAL

            # re-allocate failed tasks
            for task_item in finished_not_processed_tasks:
                if not task_item[3]:
                    self.task_queue.append(task_item[1])
                    logging.info(f"re-allocating failed task {task_item}")
                    # TODO delete task item from DB

            for task_item in finished_not_processed_tasks:
                # "process" the finished task, which is
                # decrementing indegrees of tasks that are dependent on that task
                task_id = task_item[1]
                learner_id = self.tasks[task_id].learner_id
                neighbor_ids = self.tasks[task_id].neighbor_id_list
                self.processed_tasks.append(task_id)

                # get end time of the task
                if rt_mode and not task_item[4]:
                    elasped_time = float(task_item[5]) + self.communication_time
                    freed_time = self.tasks[task_id].start + elasped_time
                elif rt_mode and task_item[4]:
                    self.timed_out_tasks += 1
                    freed_time = self.tasks[task_id].start
                else:  # if discrete mode
                    freed_time = self.tasks[task_id].start + self.communication_time + self.computation_time

                self.last_avail[learner_id] = max(self.last_avail[learner_id], freed_time)
                self.successful_tasks += 1
                # if self.dependant_to_mutable:
                # for nid in neighbor_ids:
                #     self.last_avail[nid] = max(self.last_avail[nid], freed_time)

                for next_task in self.dep_graph[task_id]:
                    # print(f"next task {next_task}")
                    self.indegrees[next_task] -= 1
                    if rt_mode:
                        self.tasks[next_task].real_time_mode = True
                        self.tasks[next_task].reset_start_time(self.last_avail)
                            
                    if self.indegrees[next_task] <= 0 \
                        and next_task not in self.processed_tasks\
                        and next_task not in self.deployed_tasks:
                        self.tasks[next_task].determine_skip()
                        self.task_queue.append(next_task)

                self.tasks_db.mark_processed(task_id)
                try:
                    self.deployed_tasks.remove(task_id)
                except:
                    logging.error(f"not deployed or already processed task {task_id} has been processed again")

            # get "Stopped" workers and check if one of them holds recent device state
            # worker_dbs = [WorkerInDB(self.swarm_name, self.worker_namespace, id) for id, ip in enumerate(self.worker_nodes)]
            # free_workers = [worker.worker_id for worker in worker_dbs if worker.status != RUNNING]
            task_id_to_worker = {}
            
            free_workers = self.worker_db.get_stopped_workers()
            

            logging.info(f"free workers {sorted(free_workers)}, task queue size {len(self.task_queue)}")
        
            if len(self.task_queue) == 0:
                blocked_time += polling_interval
                if blocked_time > 300:
                    for dt in self.deployed_tasks:
                        if dt not in self.processed_tasks:
                            self.task_queue.append(dt)
                    tset = set(self.task_queue)
                    self.task_queue = list(tset)
                    logging.info(f"re-added blocking tasks {self.task_queue}")
            else:
                blocked_time = 0

            # check if any of the instances got restarted, if they are, re-allocate the tasks
            running_workers = self.worker_db.get_running_workers()
            for w in running_workers:
                try:
                    is_init = self.send_check_initialized_request(w)
                except:
                    is_init = True
                    logging.info(f"worker {w} is down")

                if not is_init and w in self.cur_worker_to_task_id:
                    self.worker_db.update_status(w, STOPPED)
                    self.initializer._initialize_worker(self.swarm_name, w)
                    task_id = self.cur_worker_to_task_id[w]
                    self.task_queue.append(task_id)
                    self.cur_worker_to_task_id.pop(w)
                    logging.error(f"worker {w} has been restarted, task {task_id} has been re-added to queue")
               

            ## start assigning tasks to worker nodes (populate task_id_to_worker)
            # first assign based on cached device state
            # print(f"cached: {self.cached_devices_to_worker_nodes}")
            if self.learning_scenario == "oppcl":
                for task_id in self.task_queue:
                    # @TODO support mutable neighbors
                    if self.tasks[task_id].learner_id in self.cached_devices_to_worker_nodes and \
                        task_id not in task_id_to_worker and \
                        self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id] in free_workers:  
                        target_worker_id = self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id]
                        task_id_to_worker[task_id] = target_worker_id
                        free_workers.remove(target_worker_id)
                        logging.info(f"using {target_worker_id} to reuse state {self.tasks[task_id].learner_id} in {task_id}")
            
            # delete assigned tasks from task queue
            for task_id in task_id_to_worker:
                self.task_queue.remove(task_id)

            # assign remaining tasks to "Stopped" worker nodes
            while len(free_workers) > 0 and len(self.task_queue) > 0:
                worker_id = free_workers.pop()
                task_to_deploy = self.task_queue.pop()
                task_id_to_worker[task_to_deploy] = worker_id

            # print(f"{task_id_to_worker}")
            # print(f"{self.task_queue}")
            # logging.info(f"tasks left: {self.task_num}")

            # call RunTask asynchronously 
            for task_id in task_id_to_worker:
                task_thread = threading.Thread(target=self.send_run_task_request, args=(task_id_to_worker[task_id], self.tasks[task_id]))
                task_thread.start()
                self.deployed_tasks.append(task_id)
                self.allocated_tasks[task_id_to_worker[task_id]] = task_id
                self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id] = task_id_to_worker[task_id]
                self.cur_worker_to_task_id[task_id_to_worker[task_id]] = task_id

            sleep(polling_interval)
            elasped_swarm_time = time.time() - run_swarm_start_time
            logging.info(f'elasped time: {elasped_swarm_time}, remaining time: {elasped_swarm_time/(len(self.processed_tasks)+1)*(len(self.tasks)-len(self.processed_tasks))}')
        
        self.save_checkpoint({'elasped_time': f'{time.time() - run_swarm_start_time}'}, 'last')
        logging.info(f"Overmind run finished successfully with {iterations} iterations, elasped time {time.time() - run_swarm_start_time} sec.")
    
    def build_dep_graph_multi_neighbors(self, rt_mode=False, dependency=None, oppcl_time=None, time_scale=1):
        # read encounter dataset
        enc_dataset_filename = self.device_config['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)
        self.enc_df = enc_df
        last_end_time = 0
        last_run_time = 0
        
        # get dependency info
        if dependency is None:
            device_class = get_device_class(self.device_config["device_strategy"])
            dependency = device_class.get_dependency()
        self.dependant_to_mutable = dependency["on_mutable"]

        # read connection data and populate task list
        task_id = 0
        encounter_config = self.device_config['encounter_config']
        self.model_send_time = self.device_config['model_size_in_bits'] / encounter_config['communication_rate']
        self.computation_time = encounter_config['computation_time']
        if 'communication_time' not in self.device_config:
            self.communication_time = 2 * self.model_send_time 
        else:
            self.communication_time = self.device_config['communication_time']
        if not rt_mode:
            oppcl_time = 2 * self.model_send_time + self.computation_time
        else:
            oppcl_time = 0
        print(f"oppcl time: {oppcl_time}")

        dep_graph = {}  # (preceding task id, task id)
        last_tasks = dict.fromkeys(range(self.number_of_devices), set())  # (device id, last task idx) 
        last_times = dict.fromkeys(range(self.number_of_devices), 0)  # (device id, last time device is done with oppcl)
        tasks = {}  # (task id, task object)
        indegrees = {}
        # @TODO support hierarchical by specifying servers in config
        self.server_list = [0]

        for index, row in enc_df.iterrows():
            start_time = row[TIME_START] * time_scale
            end_time = row[TIME_END] * time_scale

            device1_id = (int)(row[CLIENT1])
            
            if type(row[CLIENT2]) == type([]):
                device2_ids = [int(c) for c in row[CLIENT2]]
                max_device2_id = max(device2_ids)
                is_same = (device1_id in device2_ids)
            else:
                device2_ids = [int(row[CLIENT2])]
                max_device2_id = device2_ids[0]
                is_same = (device1_id == device2_ids[0])

            if max(device1_id, max_device2_id) >= self.number_of_devices or is_same:
                continue

            if not rt_mode:
                start_time_1 = max(start_time, last_times[device1_id])
                device2_last_time = max(last_times[d2id] for d2id in device2_ids)
                start_time_2 = max(start_time, device2_last_time)

                end_time_1 = start_time_1 + oppcl_time
                end_time_2 = start_time_2 + oppcl_time
            else:
                start_time_1 = start_time
                start_time_2 = start_time
                end_time_1 = end_time
                end_time_2 = end_time

            task_1_timeout = end_time - start_time_1 < oppcl_time
            task_2_timeout = end_time - start_time_2 < oppcl_time

            if (not task_1_timeout) and (not task_2_timeout):
                task_1_2 = Task(self.swarm_name, task_id, start_time_1, end_time_1, device1_id, device2_ids, 
                                real_time_mode=rt_mode, communication_time=self.communication_time,
                                learning_config='fl')
                task_2_1 = Task(self.swarm_name, task_id+1, start_time_2, end_time_2, device2_ids[0], [device1_id], 
                                  real_time_mode=rt_mode, communication_time=self.communication_time,
                                  learning_config='fl')


                if task_1_2.learner_id not in self.server_list:  # if client
                    task_1_2.set_learner_load_config(learner_load_model=True, learner_load_dataset=True)
                    task_1_2.set_neighbor_load_config(neighbor_load_model=True, neighbor_load_dataset=False)
                    task_1_2.set_load_config()
                    for func_config in self.device_config["encounter_config"]["invoked_functions"]:
                        if func_config["func_name"][0] == '!':
                            task_1_2.add_eval()
                        else:
                            task_1_2.add_func(func_config["func_name"], func_config["params"])
                else:
                    task_1_2.set_learner_load_config(learner_load_model=True, learner_load_dataset=True)
                    task_1_2.set_neighbor_load_config(neighbor_load_model=True, neighbor_load_dataset=False)
                    task_1_2.set_load_config()
                    for func_config in self.device_config["device_groups"][0]["device_config"]["encounter_config"]["invoked_functions"]:
                        if func_config["func_name"][0] == '!':
                            task_1_2.add_eval()
                        else:
                            task_1_2.add_func(func_config["func_name"], func_config["params"])

                if not task_1_timeout:
                    indegrees[task_id] = 0
                    dep_graph[task_id] = []
                    tasks[task_id] = task_1_2

                # we assume that all delegations are dependent on data, at least
                if (not task_1_timeout):
                    if device1_id in self.server_list:
                        for lt in last_tasks[device1_id]:
                            dep_graph[lt].append(task_1_2.task_id)
                            indegrees[task_1_2.task_id] += 1
                        last_tasks[device1_id] = set()
                    else:
                        for lt in last_tasks[device1_id]:
                            dep_graph[lt].append(task_1_2.task_id)
                            indegrees[task_1_2.task_id] += 1

                    for device2_id in device2_ids:
                        if device2_id not in self.server_list:
                            for lt in last_tasks[device2_id]:
                                dep_graph[lt].append(task_1_2.task_id)
                                indegrees[task_1_2.task_id] += 1
                
                if not task_1_timeout:
                    if device1_id not in self.server_list:
                        last_tasks[device1_id] = set([task_1_2.task_id])
                    for device2_id in device2_ids:
                        last_tasks[device2_id] = set([task_1_2.task_id])


                if not rt_mode and not task_1_timeout:
                    last_times[device1_id] = max(last_times[device1_id], start_time_1 + oppcl_time)

                # if task is dependent on mutable state of the neighbor,
                # succeeding tasks should also be dependent on tasks where the device
                # participated as a neighbor
                if dependency["on_mutable"] and self.learning_scenario == "oppcl":
                    last_tasks[device1_id].append(task_2_1.task_id)

                # if learning is "mutual" not one-way, 
                if self.learning_scenario == "oppcl":  # @TODO generalize to mutual vs. one-way
                    for func_config in self.device_config["encounter_config"]["invoked_functions"]:
                        if func_config["func_name"][0] == '!':
                            task_2_1.add_eval()
                        else:
                            task_2_1.add_func(func_config["func_name"], func_config["params"])
                
                    if not task_2_timeout:
                        indegrees[task_id+1] = 0
                        dep_graph[task_id+1] = []
                        tasks[task_id+1] = task_2_1

                    for device2_id in device2_ids:
                        if (not task_2_timeout) and device2_id in last_tasks:
                            for lt in last_tasks[device2_id]:
                                dep_graph[lt].append(task_2_1.task_id)
                                indegrees[task_2_1.task_id] += 1

                        if dependency["on_mutable"]:
                            for lt in last_tasks[device1_id]:
                                dep_graph[lt].append(task_2_1.task_id)
                                indegrees[task_2_1.task_id] += 1
              
                    if not task_2_timeout:
                        for device2_id in device2_ids:
                            last_tasks[device2_id] = [task_2_1.task_id]

                    if dependency["on_mutable"]:
                        for device2_id in device2_ids:
                            last_tasks[device2_id].append(task_1_2.task_id)

                    if not rt_mode and not task_2_timeout:
                        for device2_id in device2_ids:
                            last_times[device2_id] = max(last_times[device2_id], start_time_2 + oppcl_time)

                task_id += (1 + (1 if self.learning_scenario == "oppcl" else 0))

        self.task_num = task_id
        self.dep_graph = dep_graph
        self.last_tasks = last_tasks
        self.tasks = tasks
        self.indegrees = indegrees


    def build_dep_graph(self, rt_mode=False, dependency=None, oppcl_time=None, time_scale=1):
        # read encounter dataset
        enc_dataset_filename = self.device_config['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)
        self.enc_df = enc_df
        last_end_time = 0
        last_run_time = 0
        
        # get dependency info
        if dependency is None:
            device_class = get_device_class(self.device_config["device_strategy"])
            dependency = device_class.get_dependency()
        self.dependant_to_mutable = dependency["on_mutable"]

        # read connection data and populate task list
        task_id = 0
        encounter_config = self.device_config['encounter_config']
        self.model_send_time = self.device_config['model_size_in_bits'] / encounter_config['communication_rate']
        self.computation_time = encounter_config['computation_time']
        if 'communication_time' not in self.device_config:
            self.communication_time = 2 * self.model_send_time 
        else:
            self.communication_time = self.device_config['communication_time']
        if oppcl_time == None:
            if not rt_mode:
                oppcl_time = self.communication_time + self.computation_time
            else:
                oppcl_time = 0.0000001
        print(f"oppcl time: {oppcl_time}")

        dep_graph = {}  # (preceding task id, task id)
        last_tasks = dict.fromkeys(range(self.number_of_devices), [])  # (device id, last task idx) 
        last_times = dict.fromkeys(range(self.number_of_devices), 0)  # (device id, last time device is done with oppcl)
        tasks = {}  # (task id, task object)
        indegrees = {}
        for index, row in enc_df.iterrows():
            time_start = row[TIME_START] * time_scale
            time_end = row[TIME_END] * time_scale
            device1_id = (int)(row[CLIENT1])
            device2_id = (int)(row[CLIENT2])
            if max(device1_id, device2_id) >= self.number_of_devices or device1_id == device2_id:
                continue
            start_time = max(time_start, last_times[device1_id], last_times[device2_id])
            end_time_1 = time_end
            end_time_2 = time_end
            task_timeout = time_end - start_time < oppcl_time

            if not task_timeout:
                task_1_2 = Task(self.swarm_name, task_id, start_time, end_time_1, device1_id, [device2_id],
                                real_time_mode=rt_mode, communication_time=self.communication_time, 
                                computation_time=self.computation_time, orig_enc_idx=index, bucket=self.bucket)
                task_2_1 = Task(self.swarm_name, task_id+1, start_time, end_time_2, device2_id, [device1_id],
                                real_time_mode=rt_mode, communication_time=self.communication_time,
                                computation_time=self.computation_time, orig_enc_idx=index, bucket=self.bucket)
                if "invoked_functions" in self.device_config["encounter_config"]:
                    invoked_functions = self.device_config["encounter_config"]["invoked_functions"]
                    for f in invoked_functions:
                        if f["func_name"][0] != '!':
                            task_1_2.add_func(f["func_name"], f["params"])
                            task_2_1.add_func(f["func_name"], f["params"])
                        else:
                            task_1_2.add_eval()
                            task_2_1.add_eval()
                            
                else:
                    task_1_2.add_func("delegate", {"epoch": 1, "iteration": 1})
                    task_2_1.add_func("delegate", {"epoch": 1, "iteration": 1})
                    task_1_2.add_eval()
                    task_2_1.add_eval()
                
                if dependency["on_mutable"]:
                    task_1_2.set_learner_load_config(True, True)
                    task_1_2.set_neighbor_load_config(True, False)
                    task_1_2.set_load_config()
                    task_2_1.set_learner_load_config(True, True)
                    task_2_1.set_neighbor_load_config(True, False)
                    task_2_1.set_load_config()
                
                indegrees[task_id] = 0
                dep_graph[task_id] = []
                tasks[task_id] = task_1_2

                indegrees[task_id+1] = 0
                dep_graph[task_id+1] = []
                tasks[task_id+1] = task_2_1

                # we assume that all delegations are dependent on data, at least
                if device1_id in last_tasks:
                    for lt in last_tasks[device1_id]:
                        dep_graph[lt].append(task_1_2.task_id)
                        indegrees[task_1_2.task_id] += 1

                    if dependency["on_mutable"]:
                        for lt in last_tasks[device2_id]:
                            dep_graph[lt].append(task_1_2.task_id)
                            indegrees[task_1_2.task_id] += 1

                if device2_id in last_tasks:
                    for lt in last_tasks[device2_id]:
                        dep_graph[lt].append(task_2_1.task_id)
                        indegrees[task_2_1.task_id] += 1

                    if dependency["on_mutable"]:
                        for lt in last_tasks[device1_id]:
                            dep_graph[lt].append(task_2_1.task_id)
                            indegrees[task_2_1.task_id] += 1

                last_tasks[device1_id] = [task_1_2.task_id]
                last_tasks[device2_id] = [task_2_1.task_id]
                # if task is dependent on mutable state of the neighbor,
                # succeeding tasks should also be dependent on tasks where the device
                # participated as a neighbor
                if dependency["on_mutable"]:
                    last_tasks[device1_id].append(task_2_1.task_id)
                    last_tasks[device2_id].append(task_1_2.task_id)

                task_id += 2
                last_times[device1_id] = max(last_times[device1_id], start_time + oppcl_time)
                last_times[device2_id] = max(last_times[device2_id], start_time + oppcl_time)

        self.task_num = task_id
        self.dep_graph = dep_graph
        self.last_tasks = last_tasks
        self.tasks = tasks
        self.indegrees = indegrees
    
    def send_run_task_request(self, worker_id, task):
        # task.reset_start_time(self.last_avail)
        # if task.is_skipped():  # possible error here
        #     self.processed_tasks.append(task.task_id)
        #     for next_task in self.dep_graph[task.task_id]:
        #         self.indegrees[next_task] -= 1
        #         self.tasks[next_task].reset_start_time(self.last_avail)
        #     return
        try:
            with grpc.insecure_channel(self.worker_id_to_ip[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
                stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
                config = simulate_device_pb2.Config(config=json.dumps(task.get_config()))
                status = stub.RunTask.future(config)
                res = status.result()
        except Exception as e:
            logging.error(f"gRPC call to {worker_id} returned with error")
            self.initializer._initialize_worker(self.swarm_name, worker_id)
            traceback.print_stack()
    
    def send_check_running_request(self, worker_id):
        try:
            with grpc.insecure_channel(self.worker_id_to_ip[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
                stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
                status = stub.CheckRunning(simulate_device_pb2.Empty())
                return status.status == RUNNING
        except Exception as e:
            logging.error(f"gRPC call to {worker_id} returned with error")
            traceback.print_stack()
            return False
    
    def send_check_initialized_request(self, worker_id):
        with grpc.insecure_channel(self.worker_id_to_ip[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
            stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
            status = stub.CheckInitialized(simulate_device_pb2.Empty())
            return status.status == INITIALIZED

    def revive_worker(self, worker_in_db, worker_id):
        if self.send_check_running_request(worker_id):
            worker_in_db.update_status(STOPPED)
            self.initializer._initialize_worker(self.swarm_name, worker_id)
    
    def run_swarm(self, polling_interval=5, rt_mode=False, use_cache=True):
        CHECKPOINT_INTERVAL = 1000
        ## !!Warning, rt_mode here has an error
        logging.info("----- swarm run start -----")
        run_swarm_start_time = time.time()
        self.cached_devices_to_worker_nodes = {}  # (cached_device_id, worker node)
        self.cached_worker_nodes_to_devices = {}
        self.task_queue = []
        self.processed_tasks = []
        self.deployed_tasks = []
        self.next_checkpoint = CHECKPOINT_INTERVAL
        blocked_time = 0
        for task_id in self.tasks:
            if self.indegrees[task_id] == 0:
                if rt_mode:
                    self.tasks[task_id].real_time_mode = True
                self.task_queue.append(task_id)

        worker_last_avail = {}  # last time that a worker was "idle"
        for wi in self.worker_db.worker_db_ids:
            worker_last_avail[wi] = 0

        iterations = 0
        self.successful_tasks = 0
        self.timed_out_tasks = 0

        self.allocated_tasks = {}
        self.last_avail = dict.fromkeys(range(self.number_of_devices), 0)
        self.cur_worker_to_task_id = {}

        # @TODO store which task is allocated to which and re-allocate when timeout
        while self.task_num > 0 and iterations < MAX_ITERATIONS:
            if not Path(self.log_path).exists():  # exit simulation when log file is deleted
                return

            iterations += 1

            finished_not_processed_tasks = self.tasks_db.get_not_processed_finished_tasks()
            # print(f"not fin: {finished_not_processed_tasks}")
            # checkpointing
            if len(self.processed_tasks) >= self.next_checkpoint:
                self.save_checkpoint({'elasped_time': f'{time.time() - run_swarm_start_time}'}, len(self.processed_tasks))
                self.next_checkpoint += CHECKPOINT_INTERVAL

            # re-allocate failed tasks
            for task_item in finished_not_processed_tasks:
                if not task_item[3]:
                    self.task_queue.append(task_item[1])
                    logging.info(f"re-allocating failed task {task_item}")
                    # TODO delete task item from DB

            for task_item in finished_not_processed_tasks:
                # "process" the finished task, which is
                # decrementing indegrees of tasks that are dependent on that task
                task_id = task_item[1]
                learner_id = self.tasks[task_id].learner_id
                neighbor_ids = self.tasks[task_id].neighbor_id_list
                self.processed_tasks.append(task_id)

                # get end time of the task
                if rt_mode and not task_item[4]:
                    elasped_time = float(task_item[5]) + self.communication_time
                    freed_time = self.tasks[task_id].start + elasped_time
                elif rt_mode and task_item[4]:
                    self.timed_out_tasks += 1
                    freed_time = self.tasks[task_id].start
                else:  # if discrete mode
                    freed_time = self.tasks[task_id].start + self.communication_time + self.computation_time

                self.last_avail[learner_id] = max(self.last_avail[learner_id], freed_time)
                self.successful_tasks += 1
                # if self.dependant_to_mutable:
                # for nid in neighbor_ids:
                #     self.last_avail[nid] = max(self.last_avail[nid], freed_time)

                for next_task in self.dep_graph[task_id]:
                    # print(f"next task {next_task}")
                    self.indegrees[next_task] -= 1
                    if rt_mode:
                        self.tasks[next_task].real_time_mode = True
                        self.tasks[next_task].reset_start_time(self.last_avail)
                            
                    if self.indegrees[next_task] <= 0 \
                        and next_task not in self.processed_tasks\
                        and next_task not in self.deployed_tasks:
                        self.tasks[next_task].determine_skip()
                        self.task_queue.append(next_task)

                self.tasks_db.mark_processed(task_id)
                try:
                    self.deployed_tasks.remove(task_id)
                except:
                    logging.error(f"not deployed or already processed task {task_id} has been processed again")
                self.task_num -= 1

            # get "Stopped" workers and check if one of them holds recent device state
            # worker_dbs = [WorkerInDB(self.swarm_name, self.worker_namespace, id) for id, ip in enumerate(self.worker_nodes)]
            # free_workers = [worker.worker_id for worker in worker_dbs if worker.status != RUNNING]
            task_id_to_worker = {}
            
            free_workers = self.worker_db.get_stopped_workers()
            

            logging.info(f"free workers {sorted(free_workers)}, task queue size {len(self.task_queue)}")
        
            if len(self.task_queue) == 0:
                blocked_time += polling_interval
                if blocked_time > 600:
                    for dt in self.deployed_tasks:
                        if dt not in self.processed_tasks:
                            self.task_queue.append(dt)
                    logging.info(f"re-added blocking tasks {self.task_queue}")
            else:
                blocked_time = 0

            # check if any of the instances got restarted, if they are, re-allocate the tasks
            running_workers = self.worker_db.get_running_workers()
            for w in running_workers:
                try:
                    is_init = self.send_check_initialized_request(w)
                except:
                    is_init = True
                    logging.info(f"worker {w} is down")

                if not is_init and w in self.cur_worker_to_task_id:
                    self.worker_db.update_status(w, STOPPED)
                    self.initializer._initialize_worker(self.swarm_name, w)
                    task_id = self.cur_worker_to_task_id[w]
                    self.task_queue.append(task_id)
                    self.cur_worker_to_task_id.pop(w)
                    logging.error(f"worker {w} has been restarted, task {task_id} has been re-added to queue")
               

            ## start assigning tasks to worker nodes (populate task_id_to_worker)
            # first assign based on cached device state
            # print(f"cached: {self.cached_devices_to_worker_nodes}")
            if self.learning_scenario == "oppcl" and use_cache:
                for task_id in self.task_queue:
                    # @TODO support mutable neighbors
                    if self.tasks[task_id].learner_id in self.cached_devices_to_worker_nodes and \
                        task_id not in task_id_to_worker and \
                        self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id] in free_workers:  
                        target_worker_id = self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id]
                        task_id_to_worker[task_id] = target_worker_id
                        free_workers.remove(target_worker_id)
                        logging.info(f"using {target_worker_id} to reuse state {self.tasks[task_id].learner_id} in {task_id}")
            
            # delete assigned tasks from task queue
            for task_id in task_id_to_worker:
                self.task_queue.remove(task_id)

            # assign remaining tasks to "Stopped" worker nodes
            while len(free_workers) > 0 and len(self.task_queue) > 0:
                worker_id = free_workers.pop()
                task_to_deploy = self.task_queue.pop()
                task_id_to_worker[task_to_deploy] = worker_id

            # print(f"{task_id_to_worker}")
            # print(f"{self.task_queue}")
            logging.info(f"tasks left: {self.task_num}")

            # call RunTask asynchronously 
            for task_id in task_id_to_worker:
                task_thread = threading.Thread(target=self.send_run_task_request, args=(task_id_to_worker[task_id], self.tasks[task_id]))
                task_thread.start()
                self.deployed_tasks.append(task_id)
                self.allocated_tasks[task_id_to_worker[task_id]] = task_id
                self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id] = task_id_to_worker[task_id]
                self.cur_worker_to_task_id[task_id_to_worker[task_id]] = task_id

            sleep(polling_interval)
            elasped_swarm_time = time.time() - run_swarm_start_time
            logging.info(f'elasped time: {elasped_swarm_time}, remaining time: {elasped_swarm_time/(len(self.processed_tasks)+1)*(len(self.tasks)-len(self.processed_tasks))}')
        
        self.save_checkpoint({'elasped_time': f'{time.time() - run_swarm_start_time}'}, 'last')
        logging.info(f"Overmind run finished successfully with {iterations} iterations, elasped time {time.time() - run_swarm_start_time} sec.")

    def save_checkpoint(self, log_dict, checkpoint_num):
        # save data for current progress, which are the following
        # device states
        # finished tasks
        # elasped time (which is given in log_dict)

        # save device states
        self._save_dynamodb_table(self.swarm_name, 'device_table', checkpoint_num)

        # save finished tasks
        self._save_rds_table(self.initializer.rds_tasks_cursor, 'tasks_table', checkpoint_num)

        # save log_dict and other values
        log_dict['config'] = self.config
        log_dict['run_swarm'] = {}
        log_dict['run_swarm']['tasks'] = self.tasks
        log_dict['run_swarm']['task_queue'] = self.task_queue
        log_dict['run_swarm']['processed_tasks'] = self.processed_tasks
        log_dict['run_swarm']['indegrees'] = self.indegrees
        log_dict['run_swarm']['dep_graph'] = self.dep_graph
        log_dict['run_swarm']['deployed_tasks'] = self.deployed_tasks
        log_dict['run_swarm']['checkpoint_num'] = checkpoint_num
        log_dict['run_swarm']['task_num'] = self.task_num
        

        filepath = self.log_path + f'/etc_{checkpoint_num}.log'
        with open(filepath, 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    # def load_from_checkpoint(self, )

    def _save_dynamodb_table(self, table_name, pickle_file_name, checkpoint_num):
        table = self.dynamodb.Table(table_name)
        resp = table.scan()
        filepath = self.log_path + f'/{pickle_file_name}_{checkpoint_num}.log'
        self._save_as_pickle(resp, filepath)

    def _save_rds_table(self, cursor, filename, checkpoint_num):
        record = cursor.get_all_records()
        filepath = self.log_path + f'/rds_{filename}_{checkpoint_num}.log'
        self._save_as_pickle(record, filepath)

    def _save_as_pickle(self, obj, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
