import sys
import unittest
import json
import logging
import time

from dist_swarm import DistSwarm

class DistSwarmTests(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.swarm = DistSwarm('../configs/dist_swarm/droppcl_svhn.json', 
                                    'a452d4ac8849745028e2643888ec8db8-1181542245.us-east-2.elb.amazonaws.com:80')
    
    # def test_empty(self):
    #     pass

    def test_run(self):
        time.sleep(10)
        self.swarm.run()

if __name__ == '__main__':
    unittest.main()