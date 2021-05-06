from .base_student import *

from agent import CUDAAgent
from network import StudentNetwork


class EDMStudent(BaseStudent, CUDAAgent):
    def __init__(self,
                 env: gym.Env,
                 trajs_path: str,
                 model_path: str,
                 run_seed: int,
                 batch_size: int,
                 buffer_size_in_trajs: int,
                 teacher: BaseAgent,
                 qvalue_function: StudentNetwork,
                 adam_alpha: float,
                 adam_betas: List[float],
                 sgld_buffer_size: int,
                 sgld_learn_rate: float,
                 sgld_noise_coef: float,
                 sgld_num_steps: int,
                 sgld_reinit_freq: float,
                 ):
        super(EDMStudent, self).__init__(
            env=env,
            trajs_path=trajs_path,
            model_path=model_path,
            run_seed=run_seed,
            batch_size=batch_size,
            buffer_size_in_trajs=buffer_size_in_trajs,
            teacher=teacher,
        )

        self.qvalue_function = qvalue_function.to(self.device)
        self.adam_alpha = adam_alpha
        self.adam_betas = adam_betas

        self.optimizer = optim.Adam(qvalue_function.parameters(),
                                    lr=self.adam_alpha,
                                    betas=self.adam_betas,
                                    )

        self.sgld_buffer = self._get_random_states(sgld_buffer_size)
        self.sgld_learn_rate = sgld_learn_rate
        self.sgld_noise_coef = sgld_noise_coef
        self.sgld_num_steps = sgld_num_steps
        self.sgld_reinit_freq = sgld_reinit_freq

    def select_action(self,
                      state: np.ndarray,
                      ) -> np.ndarray:
        action = self.qvalue_function(torch.FloatTensor(state).to(self.device)).argmax()
        action = action.detach().cpu().numpy()

        return action

    def train(self,
              num_updates: int,
              ):
        for _ in tqdm(range(num_updates)):
            samples = self.buffer.sample()

            loss = self._compute_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.env.close()

    def _compute_loss(self,
                      samples: Dict[str, np.ndarray],
                      ) -> torch.Tensor:
        loss_pi = self._compute_ce_loss(samples)

        state_p = torch.FloatTensor(samples['state']).to(self.device)
        state_q = self._sample_via_sgld()

        logsumexp_f_p = self.qvalue_function(state_p).logsumexp(1).mean()
        logsumexp_f_q = self.qvalue_function(state_q).logsumexp(1).mean()

        loss_rho = logsumexp_f_q - logsumexp_f_p

        loss = loss_pi + loss_rho + loss_rho ** 2

        return loss

    def _initialize_sgld(self) -> Tuple[torch.Tensor, List[int]]:
        indices = torch.randint(0,
                                len(self.sgld_buffer),
                                (self.batch_size,),
                                )

        buffer_samples = self.sgld_buffer[indices]
        random_samples = self._get_random_states(self.batch_size)

        mask = (torch.rand(self.batch_size) < self.sgld_reinit_freq).float()[:, None]
        samples = (1 - mask) * buffer_samples + mask * random_samples

        return samples.to(self.device), indices

    def _sample_via_sgld(self) -> torch.Tensor:
        samples, indices = self._initialize_sgld()

        x_t = torch.autograd.Variable(samples, requires_grad=True)

        for t in range(self.sgld_num_steps):
            grad_logsumexp = torch.autograd.grad(
                self.qvalue_function(x_t).logsumexp(1).sum(),
                [x_t], retain_graph=True)[0]

            grad_term = self.sgld_learn_rate * grad_logsumexp
            rand_term = self.sgld_noise_coef * torch.randn_like(x_t)

            x_t.data += grad_term + rand_term

        samples = x_t.detach()

        self.sgld_buffer[indices] = samples.cpu()

        return samples

    def _compute_ce_loss(self,
                         samples: Dict[str, np.ndarray],
                         ) -> torch.Tensor:
        state = torch.FloatTensor(samples['state']).to(self.device)
        action = torch.LongTensor(samples['action']).to(self.device)

        qvalues = self.qvalue_function(state)

        loss = nn.CrossEntropyLoss()(qvalues, action)

        return loss

    def _get_random_states(self,
                           num_states: int,
                           ) -> torch.Tensor:
        state_dim = self.env.observation_space.shape[0]

        return torch.FloatTensor(num_states, state_dim).uniform_(-1, 1)

    def serialize(self):
        torch.save(self.qvalue_function.state_dict(), self.model_path)

    def deserialize(self):
        self.qvalue_function.load_state_dict(torch.load(self.model_path))
