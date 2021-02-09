from .trainable_agent import *

class CUDAAgent(TrainableAgent):
    def __init__(self,
        env        : gym.Env,
        trajs_path : str    ,
    ):
        super(CUDAAgent, self).__init__(
            env        = env       ,
            trajs_path = trajs_path,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
