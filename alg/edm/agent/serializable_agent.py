from .trainable_agent import *

class SerializableAgent(TrainableAgent):
    def __init__(self,
        env             : gym.Env     ,
        trajs_path      : str         ,
        model_path      : str         ,
      ):
        super(SerializableAgent, self).__init__(
            env        = env       ,
            trajs_path = trajs_path,
        )

        self.model_path = model_path

    def serialize(self):
        raise NotImplementedError

    def deserialize(self):
        raise NotImplementedError
