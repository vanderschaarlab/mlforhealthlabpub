from .__open__           import *
from .serializable_agent import *

class OABaselineAgent(SerializableAgent):
    def __init__(self,
        env             : gym.Env,
        trajs_path      : str    ,
        model_path      : str    ,
        algorithm       : str    ,
        network         : str    ,
        num_transitions : int    ,
    ):
        super(OABaselineAgent, self).__init__(
            env        = env       ,
            trajs_path = trajs_path,
            model_path = model_path,
        )

        self.algorithm       = algorithm
        self.network         = network
        self.num_transitions = num_transitions

        self.args = ArgParse({
            'env'                 : self.env.unwrapped.spec.id,
            'env_type'            : None                      ,
            'seed'                : None                      ,
            'alg'                 : self.algorithm            ,
            'num_timesteps'       : self.num_transitions      ,
            'network'             : self.network              ,
            'gamestate'           : None                      ,
            'num_env'             : None                      ,
            'reward_scale'        : 1.0                       ,
            'save_path'           : self.model_path           ,
            'save_video_interval' : 0                         ,
            'save_video_length'   : 200                       ,
            'log_path'            : None                      ,
            'play'                : False                     ,
        })

    def select_action(self,
        state : np.ndarray,
    ) -> np.ndarray:
        lifted_state = np.expand_dims(state, axis = 0)
        lifted_action, _, _, _ = self.lifted_policy.step(lifted_state)

        return lifted_action[0]

    def train(self):
        self.lifted_policy, _ = oabase.train(self.args, {})

        self.serialize()

    def serialize(self):
        self.lifted_policy.save(self.model_path)

    def deserialize(self):
        load = oabase.get_learn_function(self.algorithm)

        env_type = oabase.get_env_type(self.args)
        network = oabase.get_default_network(env_type)

        self.lifted_policy = load(
            env             = self.env       ,
            network         = network        ,
            total_timesteps = 0              ,
            load_path       = self.model_path,
        )
