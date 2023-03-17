from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED

class Status():
    def __init__(self):
        self.status = IDLE

    def setStatus(self, stat):
        self.status = stat

    def getStatus(self, stat):
        return self.status