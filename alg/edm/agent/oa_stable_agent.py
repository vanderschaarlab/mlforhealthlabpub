from .__open__   import *
from .base_agent import *

class OAStableAgent(BaseAgent):
    def __init__(self,
        env        : gym.Env,
        trajs_path : str    ,
        algorithm  : str    ,
    ):
        super(OAStableAgent, self).__init__(
            env        = env       ,
            trajs_path = trajs_path,
        )

        self.algorithm = algorithm

        self.args = ArgParse({
            'env'           : self.env.unwrapped.spec.id            ,
            'algo'          : self.algorithm                        ,
            'folder'        : 'contrib/baselines_zoo/trained_agents',
            'n_timesteps'   : 1000                                  ,
            'n_envs'        : 1                                     ,
            'exp_id'        : -1                                    ,
            'verbose'       : 1                                     ,
            'no_render'     : True                                  ,
            'deterministic' : False                                 ,
            'stochastic'    : False                                 ,
            'load_best'     : False                                 ,
            'norm_reward'   : False                                 ,
            'seed'          : 0                                     ,
            'reward_log'    : ''                                    ,
            'gym_packages'  : []                                    ,
            'env_kwargs'    : {}                                    ,
        })

    def select_action(self,
        state : np.ndarray,
    ) -> np.ndarray:
        action, _ = self.model.predict(state, state = None, \
            deterministic = self.deterministic)

        return action

    def load_pretrained(self):
        oastable.set_global_seeds(self.args.seed)

        self._set_vars()

        self.env = oastable.create_test_env(self.args.env,
            n_envs        = self.args.n_envs       ,
            is_atari      = self.is_atari          ,
            stats_path    = self.stats_path        ,
            seed          = self.args.seed         ,
            log_dir       = self.log_dir           ,
            should_render = not self.args.no_render,
            hyperparams   = self.hyperparams       ,
            env_kwargs    = self.env_kwargs        ,
        )

        load_env = None if self.args.algo == 'acer' else self.env

        self.model = oastable.ALGOS[self.args.algo]. \
            load(self.model_path, env = load_env)

    def _set_vars(self):
        for env_module in self.args.gym_packages:
            importlib.import_module(env_module)

        log_path = os.path.join(self.args.folder, self.args.algo)

        assert os.path.isdir(log_path), \
            "The {} folder was not found".format(log_path)

        self.model_path = oastable.find_saved_model(self.args.algo, \
            log_path, self.args.env, load_best = self.args.load_best)

        self.is_atari   = 'NoFrameskip' in self.args.env
        self.stats_path = os.path.join(log_path, self.args.env)
        self.log_dir    = self.args.reward_log if self.args.reward_log != '' else None
        self.env_kwargs = {} if self.args.env_kwargs is None else self.args.env_kwargs
        self.hyperparams, self.stats_path = oastable.get_saved_hyperparams(
            self.stats_path, norm_reward = self.args.norm_reward, test_mode = True)

        self.deterministic = self.args.deterministic or self.args.algo in \
            ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not self.args.stochastic
