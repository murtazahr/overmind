from ast import parse
import cmd
import argparse
import json
import os
import time
from tabulate import tabulate
from pathlib import Path, PurePath
from dist_swarm import DistSwarm
import eks_controller, db_controller, k8s_controller

from sklearn import cluster

from dynamo_db import DEVICE_ID, EVAL_HIST_LOSS, EVAL_HIST_METRIC, HOSTNAME, \
            GOAL_DIST, LOCAL_DIST, DATA_INDICES, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, ENC_IDX, MODEL_INFO

from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED

# from dist_swarm.dist_swarm_controller import DistSwarmController

LOADBALANCER_IP = 'a00ffd26030dd49cba89f364b43d0321-342509579.us-east-2.elb.amazonaws.com:80'

class DistSwarmShell(cmd.Cmd):
    cmd.Cmd.prompt = "[SwarmCLI] "

    ### overriding methods
    def preloop(self):
        # self.controller = DistSwarmController()
        
        self.print_logo()

        self.eks_ctrl = eks_controller.EKSController()
        self.k8s_ctrl = k8s_controller.K8sController(self.eks_ctrl)
        self.db_ctrl = db_controller.DBController()

        self.current_cluster = ''
        self.current_swarm = ''

        self.swarms = {}
        self.swarm_to_cfg = {}
        self.clusters = self.eks_ctrl.list_clusters()

        if len(self.clusters) > 0:
            self.current_cluster = self.clusters[0]
            self.set_prompt(self.current_cluster, self.current_swarm)
    
    ### helper methods
    def set_prompt(self, cluster_name='', swarm_name=''):
        prompt = '[SwarmCLI'
        if len(cluster_name) > 0:
            prompt += ':{}'.format(cluster_name)
        if len(swarm_name) > 0:
            prompt += ':{}'.format(swarm_name)
        prompt += '] '
        cmd.Cmd.prompt = prompt

    def print_logo(self):
        self.logo = "█▀ █░█░█ ▄▀█ █▀█ █▀▄▀█ \n▄█ ▀▄▀▄▀ █▀█ █▀▄ █░▀░█ v. 0.01"
        self.hr = '-------------------------------------'
        print(self.hr)
        print(self.logo)
        print(self.hr)

    ### commands
    def emptyline(self):
        print('enter a valid command')

    def do_clear(self, inp):
        os.system('clear')
        self.print_logo()

    def do_create(self, inp):
        """
        create (config file for swarm, swarm)
        """
        args = inp.split(' ')
        if len(args) < 1:
            print('specify what to create (create config, create swarm)')
        if args[0] == 'config':
            self.create_config(args[1:])
        elif args[0] == 'swarm':
            self.create_swarm(args[1:])
        else:
            print('invalid command: {}'.format('create ' + args[0]))

    def do_switch(self, inp):
        args = inp.split(' ')
        if len(args) < 1:
            print('specify arguments (switch swarm)')
        if args[0] == 'cluster':
            self.switch_cluster(args[1:])
        elif args[0] == 'swarm':
            self.switch_swarm(args[1:])
        else:
            print('invalid command: {}'.format('switch ' + args[0]))

    def do_list(self, inp):
        args = inp.split(' ')
        if len(args) < 1:
            print('specify arguments (list swarm)')
        if args[0] == 'cluster':
            self.list_cluster()
        elif args[0] == 'swarm':
            self.list_swarm()
        elif args[0] == 'device':
            self.list_device()
        else:
            print('invalid command: {}'.format('list ' + args[0]))

    def do_run(self, inp):
        """
        run swarm
        """
        if len(self.current_swarm) < 1:
            print('switch to a swarm to run')
            return
        items = self.db_ctrl.get_items(self.current_swarm)
        device_states = sorted([item[HOSTNAME]['S'] for item in items], key=lambda x: x[0])
        if RUNNING in device_states:
            print('swarm {} is still running'.format(self.current_swarm))
            return
        print('running swarm {}'.format(self.current_swarm))
        if self.current_swarm in self.swarms:
            self.swarms[self.current_swarm].run()
        else:
            print('please initialize swarm')
        
    def do_edit(self, inp):
        """
        edit config file
        """
        if len(inp) < 1:
            print('specify what to edit (edit config)')
        if inp[0] == 'config':
            self.controller.edit_config(inp[1:])

        raise NotImplementedError()

    def do_remove_config(self, inp):
        """
        remove (config file for swarm, swarm)
        """
        if len(inp) < 1:
            print('specify what to create (remove config, remove swarm)')
        if inp[0] == 'config':
            self.controller.remove_config(inp[1:])
        elif inp[1] == 'swarm':
            self.controller.remove_swarm(inp[1:])
        else:
            print('invalid command: {}'.format('create ' + inp[0]))

    ### swarm methods    
    def create_swarm(self, inp):
        """
        create swarm from config file
        """
        parser = argparse.ArgumentParser(prog="config")
        parser.add_argument('--config', dest='config_file', 
                            default='configs/dist_swarm/controller_example.json')
        parsed = parser.parse_args(inp)

        new_swarm = DistSwarm(parsed.config_file, LOADBALANCER_IP)
        # @TODO get loadbalancer ip dynamically

        with open(parsed.config_file, 'rb') as f:
            config_json = f.read()
        config = json.loads(config_json)

        if config['tag'] in self.swarms:
            print('swarm name {} already exists'.format(config['tag']))

        self.swarms[config['tag']] = new_swarm
        self.swarm_to_cfg[config['tag']] = parsed.config_file

        print('created swarm {} with config file {}'.format(config['tag'], parsed.config_file))

    def switch_swarm(self, inp):
        if len(inp) != 1:
            print('invalid arguments. Please specify the name of a swarm')
            return
        swarms = self.db_ctrl.list_tables()
        if inp[0] in swarms:
            self.current_swarm = inp[0]
            self.set_prompt(self.current_cluster, self.current_swarm)
        else:
            print('swarm {} does not exist'.format(inp[0]))        
    
    def list_swarm(self):
        for sn in self.db_ctrl.list_tables():
            print(sn)

    def run_swarm(self):
        if len(self.current_swarm) < 1:
            print('switch into a swarm to list devices')
        try:
            self.swarms[self.current_swarm].run()
        except:
            print('running swarm failed with unknown error')

    ### devices methods
    def list_device(self):
        if len(self.current_swarm) < 1:
            print('switch into a swarm to list devices')
        items = self.db_ctrl.get_items(self.current_swarm)
        devices = sorted([[item[DEVICE_ID]['N'], item[DEV_STATUS]['S'], item[HOSTNAME]['S']] for item in items], key=lambda x: x[0])
        print(tabulate(devices, headers=['id', 'status', 'pod']))

    ### namespace methods
    def list_namespace(self):
        for sn in self.k8s_ctrl.list_namespace():
            print(sn)

    def switch_namespace(self, inp):
        if len(inp) != 1:
            print('invalid arguments. Please specify the name of a namespace')
            return
        namespaces = self.k8s_ctrl.list_namespace()
        print(inp[0])
        if inp[0] in namespaces:
            self.current_namespace = inp[0]
        else:
            print('namespace {} does not exist'.format(inp[0]))        

    def do_config_swarm(self, inp):
        """
        configure a specificed swarm
        """
        raise NotImplementedError()

    def do_set_default_swarm(self, inp):
        """
        set default swarm
        """
        raise NotImplementedError()

    def do_show_swarm(self, inp):
        """
        show specifics of swarm
        """
        raise NotImplementedError()

    def do_remove_swarm(self, inp):
        """
        remove specified swarm
        """
        raise NotImplementedError()

    ### cluster methods
    def switch_cluster(self, inp):
        if len(inp) != 1:
            print('invalid arguments. Please specify the name of a cluster')
            return
        clusters = self.eks_ctrl.list_clusters()
        print(inp[0])
        if inp[0] in clusters:
            self.current_cluster = inp[0]
            self.set_prompt(self.current_swarm, self.current_cluster)
        else:
            print('cluster {} does not exist'.format(inp[0]))        

    def list_cluster(self):
        for cn in self.eks_ctrl.list_clusters():
            print(cn)

    def do_exit(self, inp):
        return True

    ### control logics
    

def main():
    DistSwarmShell().cmdloop()

if __name__ == "__main__":
    main()