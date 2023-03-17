"""
resides in user side
"""

from kubernetes import client, config
import logging

class K8sController():
    """
    k8s controller for distributed swarm simulation. Uses AWS EKS.
    Currently, one pod per node, which is called a worker
    """
    def __init__(self, eks_controller) -> None:
        self.eks_controller = eks_controller

        # Configs can be set in Configuration class directly or using helper utility
        config.load_kube_config()

        self.api = client.CoreV1Api()
        logging.info("Listing pods with their IPs:")
        ret = self.api.list_namespaced_pod(namespace=self.eks_controller.get_namespace(), label_selector='app=dist-swarm')
        self.pods = {}
        for i in ret.items:
            self.pods[i.metadata.name] = i.status.pod_ip
            # print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
        logging.info('Total {} pods read'.format(len(self.pods)))

        ret = self.api.list_namespaced_pod(namespace=self.eks_controller.get_namespace(), label_selector='role=controller')
        for i in ret.items:
            logging.info("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

    def create_namespace(self, name):
        self.api.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=name)))
        #@TODO Error handling if duplicate name

    def list_namespace(self):
        ret = self.api.list_namespace()
        names = []
        
        for i in ret.items:
            if i.metadata.name.split('-')[-1] == 'namespace':
                names.append('{}'.format(('-').join(i.metadata.name.split('-')[:-1])))

        return names

    def update_pods(self, container_hash):
        """
        update pods with a docker container 
        """
        raise NotImplementedError()


