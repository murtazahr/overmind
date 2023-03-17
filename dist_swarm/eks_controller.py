import boto3

class EKSController():
    def __init__(self):
        self.client = boto3.client('eks')
        self.clusters = []

        self.namespace = 'ds-namespace'

    def create_cluster(self, name):
        if name in self.clusters:
            print('cluster {} already exists'.format(name))
            return
        
        resp = self.client.create_cluster(name=name, version='1.21',
                                          roleArn='arn:aws:iam::115215637117:user/Sangsu',
                                          )
        print('cluster {} status: {}'.format(name, resp.cluster.status))
        # @TODO figure out how to put resourcesVpcConfig parameter
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/eks.html#EKS.Client.create_cluster
        # https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html

    def list_clusters(self):
        resp = self.client.list_clusters()
        return resp['clusters']

    def get_namespace(self):
        return self.namespace