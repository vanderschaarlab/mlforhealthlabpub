from .__head__ import *

from agent  import BaseAgent, SerializableAgent
from buffer import ReplayBuffer

class BaseStudent(SerializableAgent):
    def __init__(self,
        env                  : gym.Env  ,
        trajs_path           : str      ,
        model_path           : str      ,
        run_seed             : int      ,
        batch_size           : int      ,
        buffer_size_in_trajs : int      ,
        teacher              : BaseAgent,
    ):
        super(BaseStudent, self).__init__(
            env        = env       ,
            trajs_path = trajs_path,
            model_path = model_path,
        )

        self.run_seed             = run_seed
        self.batch_size           = batch_size
        self.buffer_size_in_trajs = buffer_size_in_trajs
        self.teacher              = teacher

        self._fill_buffer()

    def matchup(self) -> np.ndarray:
        samples = self.buffer.sample_all()
        state   = samples['state' ]
        action  = samples['action']

        action_hat = np.array([self.select_action(s) for s in state])
        match_samp =  np.equal(action, action_hat)

        return match_samp

    def rollout(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], float]:
        state = self.env.reset()

        traj = []
        match = []
        retvrn = 0

        done = False

        while not done:
            action = self.select_action(state)
            reward, next_state, done = self.perform_action(action)

            traj += [(state, action)]
            match += [action == self.teacher.select_action(state)]
            retvrn += reward

            state = next_state

        return traj, match, retvrn

    def test(self,
        num_episodes : int,
    ) -> Tuple[float, float, float]:
        self.test_mode = True

        trajs = []
        matches = []
        returns = []

        for episode_index in range(num_episodes):
            traj, match, retvrn = self.rollout()

            trajs += [traj]
            matches += match
            returns += [retvrn]

        np.save(self.trajs_path, {'trajs': trajs, 'returns': returns})

        return np.sum(matches) / len(matches), np.mean(returns), np.std(returns)

    def serialize(self):
        raise NotImplementedError

    def deserialize(self):
        raise NotImplementedError

    def _fill_buffer(self):
        trajs = np.load(self.teacher.trajs_path, allow_pickle = True)[()] \
            ['trajs'][self.run_seed:self.run_seed + self.buffer_size_in_trajs]

        pairs = [pair for traj in trajs for i, pair in enumerate(traj) if i % 20 == 0]

        if len(pairs) < self.batch_size:
            self.batch_size = len(pairs)

        self.buffer = ReplayBuffer(
            state_dim  = self.env.observation_space.shape[0],
            total_size = len(pairs)                         ,
            batch_size = self.batch_size                    ,
        )

        for pair in pairs:
            self.buffer.store(
                state      = pair[0],
                action     = pair[1],
                reward     = None   ,
                next_state = None   ,
                done       = None   ,
            )
