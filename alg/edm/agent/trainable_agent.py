from .base_agent import *

class TrainableAgent(BaseAgent):
    def train(self):
        self.test_mode = False
