import logging
import threading
import time

class ThreadEx():
    def init(self):
        self.name = 'mythread'

    def thread_function(self, name):
        logging.info("Thread %s: starting", self.name)
        time.sleep(2)
        logging.info("Thread %s: finishing", name)

    def start_test(self):
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO,
                            datefmt="%H:%M:%S")

        logging.info("Main    : before creating thread")
        x = threading.Thread(target=self.thread_function, args=(1,))
        logging.info("Main    : before running thread")
        x.start()
        logging.info("Main    : wait for the thread to finish")
        # x.join()
        logging.info("Main    : all done")

if __name__ == "__main__":
    te = ThreadEx()
    te.init()
    te.start_test()