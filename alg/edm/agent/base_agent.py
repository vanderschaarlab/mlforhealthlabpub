from .__head__ import *

class BaseAgent:
    def __init__(self,
        env        : gym.Env,
        trajs_path : str    ,
    ):
        self.env = env
        self.trajs_path = trajs_path
        self.test_mode = False

    def select_action(self,
        state : np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def perform_action(self,
        action : np.ndarray,
    ) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        return reward, next_state, done

    def rollout(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
        state = self.env.reset()

        traj = []
        retvrn = 0

        done = False

        while not done:
            action = self.select_action(state)
            reward, next_state, done = self.perform_action(action)

            traj += [(state, action)]
            retvrn += reward

            state = next_state

        return traj, retvrn

    def test(self,
        num_episodes : int,
    ) -> Tuple[float, float]:
        self.test_mode = True

        trajs = []
        returns = []

        for episode_index in range(num_episodes):
            traj, retvrn = self.rollout()

            trajs += [traj]
            returns += [retvrn]

        return np.mean(returns), np.std(returns)
