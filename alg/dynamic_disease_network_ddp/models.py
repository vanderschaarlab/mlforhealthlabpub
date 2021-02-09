import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def max_likelihood_one_step(likelihood, opt):
    loss = -1.0 * likelihood
    loss.backward()
    opt.step()
    opt.zero_grad()


def cross_ent_one_step(ent_loss, opt):
    ent_loss.backward()
    opt.step()
    opt.zero_grad()


class MvtHawkes(torch.nn.Module):
    """
    Multivariate Hawkes Process
    """

    def __init__(self, n_event_type, first_occurrence_only=False, mu_vec_np=None, alpha_mat_np=None,
                 lambda_mat_np=None):
        super().__init__()
        self.n_event_type = n_event_type
        self.dtype = torch.float32
        self.first_occurrence_only = first_occurrence_only

        self.mu_vec = None
        self.alpha_mat = None
        self.lambda_mat = None
        self._initialize_parameters(mu_vec_np, alpha_mat_np, lambda_mat_np)

        # data
        self.seq_time_to_end = None
        self.seq_time_to_current = None
        self.seq_type_event = None
        self.time_since_start_to_end = None
        self.seq_mask = None
        self.seq_mask_to_current = None
        self.intensity_mask = None
        self.event_time_to_end = None

        # temp results
        self.alpha_over_seq = None
        self.lambda_over_seq = None
        self.integral_varying = None
        self.integral_constant = None
        self.sum_log_activation = None

        # loss and eval metrics
        self.cross_ent_loss_seq = None
        self.likelihood_seq = None

    def set_input(self,
                  seq_time_to_end,
                  seq_time_to_current,
                  seq_type_event,
                  time_since_start_to_end,
                  seq_mask,
                  seq_mask_to_current,
                  intensity_mask=None,
                  event_time_to_end=None,
                  static_context=None):
        assert static_context is not None
        self.seq_time_to_end = seq_time_to_end
        self.seq_time_to_current = seq_time_to_current
        self.seq_type_event = seq_type_event
        self.time_since_start_to_end = time_since_start_to_end
        self.seq_mask = seq_mask
        self.seq_mask_to_current = seq_mask_to_current
        if self.first_occurrence_only:
            self.intensity_mask = intensity_mask
            self.event_time_to_end = event_time_to_end
        else:
            self.intensity_mask = None
            self.event_time_to_end = None

    def _initialize_parameters(self, mu_vec_np=None, alpha_mat_np=None, lambda_mat_np=None):
        if mu_vec_np is None:
            mu_vec_np = np.random.uniform(0, 1, self.n_event_type)
        if alpha_mat_np is None:
            alpha_mat_np = np.random.uniform(0, 1, (self.n_event_type, self.n_event_type))
        if lambda_mat_np is None:
            lambda_mat_np = np.random.uniform(0, 1, (self.n_event_type, self.n_event_type))

        def _map_np_to_torch_parameter(np_array):
            return nn.Parameter(torch.tensor(np_array, dtype=self.dtype, requires_grad=True))

        self.mu_vec, self.alpha_mat, self.lambda_mat = map(
            _map_np_to_torch_parameter, (mu_vec_np, alpha_mat_np, lambda_mat_np)
        )

    def _update_alpha_lambda_over_seq(self):
        seq_type_event = self.seq_type_event
        intensity_mask = self.intensity_mask

        if intensity_mask is None:
            self.alpha_over_seq = torch.exp(self.alpha_mat[:, seq_type_event])
            self.lambda_over_seq = torch.exp(self.lambda_mat[:, seq_type_event])
        else:
            self.alpha_over_seq = torch.exp(self.alpha_mat[:, seq_type_event]) * intensity_mask
            self.lambda_over_seq = torch.exp(self.lambda_mat[:, seq_type_event]) * intensity_mask

    def reg_alpha_mat_l1(self, strength=1.0):
        return torch.sum(torch.exp(self.alpha_mat)) * strength

    def _update_integral_varying(self):
        seq_time_to_end = self.seq_time_to_end
        seq_mask = self.seq_mask
        intensity_mask = self.intensity_mask
        event_time_to_end = self.event_time_to_end

        if event_time_to_end is None:
            # K, T, Batch
            time_integral = seq_time_to_end[None, :, :]
        else:
            time_integral = (seq_time_to_end[None, :, :] - event_time_to_end[:, None, :]) * intensity_mask

        term_3 = torch.sum(
            torch.sum(
                (
                        (
                                np.float32(1.0) - torch.exp(-self.lambda_over_seq * time_integral)
                        ) * self.alpha_over_seq
                ),
                dim=0
            ) * seq_mask,
            dim=0
        )
        self.integral_varying = term_3

    def _update_integral_constant(self):
        time_since_start_to_end = self.time_since_start_to_end
        event_time_to_end = self.event_time_to_end

        if event_time_to_end is None:
            term_2 = torch.sum(torch.exp(self.mu_vec)) * time_since_start_to_end
        else:
            term_2 = torch.sum(torch.exp(self.mu_vec)[:, None] * (time_since_start_to_end[None, :] - event_time_to_end))
        self.integral_constant = term_2

    def _get_constant_term(self):
        if self.mu_vec.dim() == 1:
            const_term = torch.exp(self.mu_vec)[:, None, None]
        else:
            raise ValueError('Dimension of mu_vec is not 1.')
        return const_term

    def get_prediction(self):
        seq_mask_to_current = self.seq_mask_to_current
        seq_time_to_current = self.seq_time_to_current

        assert not self.first_occurrence_only

        self._update_alpha_lambda_over_seq()
        lambda_over_seq = self._get_constant_term() + torch.sum(
            (
                    seq_mask_to_current[None, :, :, :]
                    * (
                            self.alpha_over_seq[:, None, :, :] * self.lambda_over_seq[:, None, :, :] * torch.exp(
                        -self.lambda_over_seq[:, None, :, :] * seq_time_to_current[None, :, :, :]
                    )
                    )
            )
            , dim=2
        )
        total_lambda_over_seq = torch.sum(lambda_over_seq, dim=0)
        lambda_ratio_over_seq = lambda_over_seq / total_lambda_over_seq
        return lambda_ratio_over_seq

    def next_event_time(self, n_sims):
        time_diffs = torch.empty(n_sims, dtype=self.seq_time_to_current.dtype, device=self.seq_time_to_current.device)
        nn.init.uniform_(time_diffs, 0., 5.)
        time_diffs, _ = torch.sort(time_diffs)

        seq_time_to_current = self.seq_time_to_current[-1, :, :]
        # T x batch x M
        seq_time_to_current = seq_time_to_current[None, :, :, None] + time_diffs[None, None, None, :]

        self._update_alpha_lambda_over_seq()
        # n_event_type x batch x M
        lambda_over_seq = self._get_constant_term() + torch.sum(
            (
                    self.alpha_over_seq[:, :, :, None] * self.lambda_over_seq[:, :, :, None] * torch.exp(
                -self.lambda_over_seq[:, :, :, None] * seq_time_to_current
            )
            )
            , dim=1  # sum over t
        )

        # size_batch * M: lambda_sum_each_step
        lambda_sum_each_step = torch.sum(lambda_over_seq, dim=0)

        cum_num = torch.arange(time_diffs.shape[0] + 1, device=lambda_over_seq.device)[1:] * 1.0

        # M
        term_2 = torch.exp(
            (-1.0 * torch.cumsum(lambda_sum_each_step, dim=1) / cum_num[None, :]) * time_diffs[None, :]
        )

        # size_batch * M
        term_3 = lambda_sum_each_step
        density = term_2 * term_3 + 1E-9
        # size_batch * M
        # time_prediction_each_step = torch.mean(
        #     term_1[None, :] * density, dim=1
        # ) * time_diffs[-1]

        # size_batch
        pred_time = torch.sum(time_diffs[None, :] * density, dim=1) / (torch.sum(density, dim=1))

        # T x B
        seq_time_to_current = self.seq_time_to_current[-1, :, :] + pred_time[None, :]

        # Evt x B
        lambda_next_event = self._get_constant_term()[:, :, 0] + torch.sum(
            (
                    self.alpha_over_seq * self.lambda_over_seq * torch.exp(
                -self.lambda_over_seq * seq_time_to_current[None, :, :])
            )
            , dim=1  # sum over t
        )

        total_lambda_over_seq = torch.sum(lambda_next_event, dim=0) + 1E-9
        pred_score = lambda_next_event / total_lambda_over_seq
        _, pred_event = torch.max(pred_score, dim=0)
        pred_time = pred_time + self.time_since_start_to_end
        return pred_time, pred_event, pred_score  # , lambda_over_seq, density, time_diffs

    def update_state_given_event(self, pred_time, pred_event):
        # B; Evt x B

        time_delta = pred_time - self.time_since_start_to_end
        # T + 1 x B
        self.seq_time_to_end = nn.functional.pad(self.seq_time_to_end + time_delta[None, :], (0, 0, 0, 1))

        # T + 1 x T + 1 x B
        self.seq_time_to_current = nn.functional.pad(self.seq_time_to_current, (0, 0, 0, 1, 0, 1))
        self.seq_time_to_current[-1, :, :] = self.seq_time_to_end - self.seq_time_to_end[-1, :]

        # T + 1 x B
        self.seq_type_event = torch.cat([self.seq_type_event, pred_event[None, :]], dim=0)

        # B
        self.time_since_start_to_end = pred_time

        # T + 1 x B
        self.seq_mask = nn.functional.pad(self.seq_mask, (0, 0, 0, 1), value=1.)

        # T + 1 x T + 1 x B
        self.seq_mask_to_current = nn.functional.pad(self.seq_mask_to_current, (0, 0, 0, 1, 0, 1), value=1.)
        self.seq_mask_to_current[:, -1, :] = 0.

    def get_prediction_cross_ent_loss(self):
        seq_type_event = self.seq_type_event
        seq_mask = self.seq_mask

        pred = self.get_prediction()
        mvt_pred_log = torch.log(pred).permute((2, 0, 1))
        loss_target = seq_type_event.permute((1, 0))
        nll_loss = torch.nn.functional.nll_loss(mvt_pred_log, loss_target, reduction='none').permute((1, 0))
        self.cross_ent_loss_seq = torch.sum(nll_loss * seq_mask, dim=0)
        sum_loss = torch.sum(self.cross_ent_loss_seq)
        return sum_loss

    def get_prediction_cross_ent_loss_ignore_first(self):
        seq_type_event = self.seq_type_event
        seq_mask = self.seq_mask
        seq_mask[0, :] = 0

        pred = self.get_prediction()
        mvt_pred_log = torch.log(pred).permute((2, 0, 1))
        loss_target = seq_type_event.permute((1, 0))
        nll_loss = torch.nn.functional.nll_loss(mvt_pred_log, loss_target, reduction='none').permute((1, 0))
        self.cross_ent_loss_seq = torch.sum(nll_loss * seq_mask, dim=0)
        sum_loss = torch.sum(self.cross_ent_loss_seq)
        return sum_loss

    def _update_sum_log_activation(self):
        seq_mask_to_current = self.seq_mask_to_current
        seq_time_to_current = self.seq_time_to_current
        seq_type_event = self.seq_type_event
        seq_mask = self.seq_mask

        lambda_over_seq = self._get_constant_term() + torch.sum(
            (
                    seq_mask_to_current[None, :, :, :]
                    * (
                            self.alpha_over_seq[:, None, :, :] * self.lambda_over_seq[:, None, :, :] * torch.exp(
                        -self.lambda_over_seq[:, None, :, :] * seq_time_to_current[None, :, :, :]
                    )
                    )
            )
            , dim=2
        )

        new_shape_0 = lambda_over_seq.shape[1] * lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]

        lambda_target_over_seq = lambda_over_seq.permute(
            (1, 2, 0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            np.arange(new_shape_0),
            seq_type_event.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )

        log_lambda_target_over_seq = torch.log(lambda_target_over_seq * 100.0 + 1e-9) - torch.log(torch.tensor(100.0))
        log_lambda_target_over_seq *= seq_mask

        term_1 = torch.sum(log_lambda_target_over_seq, dim=0)
        self.sum_log_activation = term_1

    def get_likelihood_seq(self):

        self._update_alpha_lambda_over_seq()
        self._update_integral_constant()
        self._update_integral_varying()
        self._update_sum_log_activation()
        likelihood_seq = self.sum_log_activation - self.integral_constant - self.integral_varying
        self.likelihood_seq = likelihood_seq
        return likelihood_seq

    def forward(self):
        likelihood_seq = self.get_likelihood_seq()
        likelihood = torch.sum(likelihood_seq)
        return likelihood


class CHawkes(MvtHawkes):
    """
    Contextual Hawkes Process
    """
    def __init__(self, n_event_type, n_context_dim, first_occurrence_only=False, mu_vec_np=None, alpha_mat_np=None,
                 lambda_mat_np=None):
        super().__init__(n_event_type, first_occurrence_only, mu_vec_np, alpha_mat_np, lambda_mat_np)
        self.lin = nn.Linear(n_context_dim, n_event_type)
        self.static_context = None

    def set_input(self,
                  seq_time_to_end,
                  seq_time_to_current,
                  seq_type_event,
                  time_since_start_to_end,
                  seq_mask,
                  seq_mask_to_current,
                  intensity_mask=None,
                  event_time_to_end=None,
                  static_context=None):
        assert static_context is not None
        self.seq_time_to_end = seq_time_to_end
        self.seq_time_to_current = seq_time_to_current
        self.seq_type_event = seq_type_event
        self.time_since_start_to_end = time_since_start_to_end
        self.seq_mask = seq_mask
        self.seq_mask_to_current = seq_mask_to_current
        self.static_context = static_context
        if self.first_occurrence_only:
            self.intensity_mask = intensity_mask
            self.event_time_to_end = event_time_to_end
        else:
            self.intensity_mask = None
            self.event_time_to_end = None
        self._update_mu()

    def _initialize_parameters(self, mu_vec_np=None, alpha_mat_np=None, lambda_mat_np=None):
        if alpha_mat_np is None:
            alpha_mat_np = np.random.uniform(0, 1, (self.n_event_type, self.n_event_type))
        if lambda_mat_np is None:
            lambda_mat_np = np.random.uniform(0, 1, (self.n_event_type, self.n_event_type))

        def _map_np_to_torch_parameter(np_array):
            return nn.Parameter(torch.tensor(np_array, dtype=self.dtype, requires_grad=True))

        self.alpha_mat, self.lambda_mat = map(
            _map_np_to_torch_parameter, (alpha_mat_np, lambda_mat_np)
        )

    def _update_mu(self):
        """
        Update the context-dependent mu vector
        :param static_context: the context tensor (n_context_dim, batch_size)
        :return: None
        """
        static_context = self.static_context
        # input to lin: batch_size * n_context_dim
        # mu_vec: n_event_type * batch_size
        self.mu_vec = torch.exp(self.lin(static_context.t())).t()

    def _update_integral_constant(self):
        time_since_start_to_end = self.time_since_start_to_end
        event_time_to_end = self.event_time_to_end

        if event_time_to_end is None:
            term_2 = torch.sum(self.mu_vec, dim=0) * time_since_start_to_end
        else:
            term_2 = torch.sum(self.mu_vec * (time_since_start_to_end[None, :] - event_time_to_end))

        self.integral_constant = term_2

    def _get_constant_term(self):
        if self.mu_vec.dim() == 2:
            # K x T x Batch_size
            const_term = self.mu_vec[:, None, :]
        else:
            raise ValueError('Dimension of mu_vec is not 2.')
        return const_term


class GraphHawkes(CHawkes):
    def __init__(self, n_event_type, n_context_dim, first_occurrence_only=False, mu_vec_np=None, alpha_mat_np=None,
                 lambda_mat_np=None, embedding_size=10, rnn_hidden_size=10):
        super().__init__(n_event_type, n_context_dim, first_occurrence_only, mu_vec_np, alpha_mat_np, lambda_mat_np)
        self.embeding = nn.Embedding(n_event_type, embedding_size)
        self.rnn = nn.LSTM(embedding_size, rnn_hidden_size)
        self.rnn_lin = nn.Linear(rnn_hidden_size, 1)
        self.graph_weights_to_current = None
        self.graph_weights_seq = None

    def set_input(self,
                  seq_time_to_end,
                  seq_time_to_current,
                  seq_type_event,
                  time_since_start_to_end,
                  seq_mask,
                  seq_mask_to_current,
                  intensity_mask=None,
                  event_time_to_end=None,
                  static_context=None):
        super().set_input(
            seq_time_to_end,
            seq_time_to_current,
            seq_type_event,
            time_since_start_to_end,
            seq_mask,
            seq_mask_to_current,
            intensity_mask,
            event_time_to_end,
            static_context)
        self._update_graph_weights()

    def _update_graph_weights(self):
        seq_type_event = self.seq_type_event
        seq_mask_to_current = self.seq_mask_to_current

        # n_event_type: T * batch_size
        # event_embedding: T * batch_size * K
        event_embedding = self.embeding(seq_type_event)
        # rnn_output: T * batch_size * rnn_hidden_size
        rnn_output, _ = self.rnn(event_embedding)
        # graph_weights: T * batch_size
        # activation function
        graph_weights = torch.sigmoid(self.rnn_lin(rnn_output)).reshape(rnn_output.shape[:2])
        self.graph_weights_seq = graph_weights
        self.graph_weights_to_current = graph_weights[None, :, :] * seq_mask_to_current

    def reg_graph_weights_l1(self, strength=1.0):
        return torch.sum(self.graph_weights_seq) * strength

    def _update_sum_log_activation(self):
        seq_mask_to_current = self.seq_mask_to_current
        seq_time_to_current = self.seq_time_to_current
        seq_type_event = self.seq_type_event
        seq_mask = self.seq_mask

        lambda_over_seq = self._get_constant_term() + torch.sum(
            (
                    seq_mask_to_current[None, :, :, :]
                    * (
                            self.alpha_over_seq[:, None, :, :] * self.graph_weights_to_current[None, :, :,
                                                                 :] * self.lambda_over_seq[:, None, :, :] * torch.exp(
                        -self.lambda_over_seq[:, None, :, :] * seq_time_to_current[None, :, :, :]
                    )
                    )
            )
            , dim=2
        )

        new_shape_0 = lambda_over_seq.shape[1] * lambda_over_seq.shape[2]
        new_shape_1 = lambda_over_seq.shape[0]
        back_shape_0 = lambda_over_seq.shape[1]
        back_shape_1 = lambda_over_seq.shape[2]

        lambda_target_over_seq = lambda_over_seq.permute(
            (1, 2, 0)
        ).reshape(
            (
                new_shape_0, new_shape_1
            )
        )[
            np.arange(new_shape_0),
            seq_type_event.flatten()
        ].reshape(
            (back_shape_0, back_shape_1)
        )

        log_lambda_target_over_seq = torch.log(lambda_target_over_seq * 100.0 + 1e-9) - torch.log(torch.tensor(100.0))
        log_lambda_target_over_seq *= seq_mask

        term_1 = torch.sum(log_lambda_target_over_seq, dim=0)
        self.sum_log_activation = term_1

    def _update_integral_varying(self):
        seq_time_to_end = self.seq_time_to_end
        seq_mask = self.seq_mask
        intensity_mask = self.intensity_mask
        event_time_to_end = self.event_time_to_end

        if event_time_to_end is None:
            # K, T, Batch
            time_integral = seq_time_to_end[None, :, :]
        else:
            time_integral = (seq_time_to_end[None, :, :] - event_time_to_end[:, None, :]) * intensity_mask

        term_3 = torch.sum(
            torch.sum(
                (
                        (
                                np.float32(1.0) - torch.exp(-self.lambda_over_seq * time_integral)
                        ) * self.alpha_over_seq * self.graph_weights_seq[None, :, :]
                ),
                dim=0
            ) * seq_mask,
            dim=0
        )
        self.integral_varying = term_3

    def get_prediction(self):
        seq_mask_to_current = self.seq_mask_to_current
        seq_time_to_current = self.seq_time_to_current
        seq_type_event = self.seq_type_event
        seq_mask = self.seq_mask

        assert not self.first_occurrence_only
        self._update_alpha_lambda_over_seq()
        lambda_over_seq = self._get_constant_term() + torch.sum(
            (
                    seq_mask_to_current[None, :, :, :]
                    * (
                            self.alpha_over_seq[:, None, :, :] * self.graph_weights_to_current[None, :, :,
                                                                 :] * self.lambda_over_seq[:, None, :, :] * torch.exp(
                        -self.lambda_over_seq[:, None, :, :] * seq_time_to_current[None, :, :, :]
                    )
                    )
            )
            , dim=2
        )  # * seq_mask
        total_lambda_over_seq = torch.sum(lambda_over_seq, dim=0)
        lambda_ratio_over_seq = lambda_over_seq / total_lambda_over_seq
        return lambda_ratio_over_seq

    def next_event_time(self, n_sims):
        time_diffs = torch.empty(n_sims, dtype=self.seq_time_to_current.dtype, device=self.seq_time_to_current.device)
        nn.init.uniform_(time_diffs, 0., 5.)
        time_diffs, _ = torch.sort(time_diffs)

        seq_time_to_current = self.seq_time_to_current[-1, :, :]
        # T x batch x M
        seq_time_to_current = seq_time_to_current[None, :, :, None] + time_diffs[None, None, None, :]

        self._update_alpha_lambda_over_seq()

        xbb = (self.alpha_over_seq[:, :, :, None] * self.graph_weights_seq[None, :, :, None] * self.lambda_over_seq[:,
                                                                                               :, :, None]) * torch.exp(
            -self.lambda_over_seq[:, :, :, None] * seq_time_to_current
        )
        # Evt x Batch_size
        const = self._get_constant_term()[:, 0, :]

        lambda_over_seq = const[:, :, None] + torch.sum(xbb, dim=1)

        # size_batch * M: lambda_sum_each_step
        lambda_sum_each_step = torch.sum(lambda_over_seq, dim=0)

        cum_num = torch.arange(time_diffs.shape[0] + 1, device=lambda_over_seq.device)[1:] * 1.0

        # M
        term_2 = torch.exp(
            (-1.0 * torch.cumsum(lambda_sum_each_step, dim=1) / cum_num[None, :]) * time_diffs[None, :]
        )

        # size_batch * M
        term_3 = lambda_sum_each_step
        density = term_2 * term_3 + 1E-9
        # size_batch * M
        # time_prediction_each_step = torch.mean(
        #     term_1[None, :] * density, dim=1
        # ) * time_diffs[-1]

        # size_batch
        pred_time = torch.sum(time_diffs[None, :] * density, dim=1) / (torch.sum(density, dim=1))

        # T x B
        seq_time_to_current = self.seq_time_to_current[-1, :, :] + pred_time[None, :]

        # Evt x B
        lambda_next_event = const + torch.sum(
            (
                    self.alpha_over_seq * self.lambda_over_seq * self.graph_weights_seq[None, :, :] * torch.exp(
                -self.lambda_over_seq * seq_time_to_current[None, :, :])
            )
            , dim=1  # sum over t
        )

        total_lambda_over_seq = torch.sum(lambda_next_event, dim=0) + 1E-9
        pred_score = lambda_next_event / total_lambda_over_seq
        _, pred_event = torch.max(pred_score, dim=0)
        pred_time = pred_time + self.time_since_start_to_end

        return pred_time, pred_event, pred_score

    def update_state_given_event(self, pred_time, pred_event):
        super(GraphHawkes, self).update_state_given_event(pred_time, pred_event)
        self._update_graph_weights()


class DDP(GraphHawkes):
    """
    DDP model
    """
    def __init__(self, n_event_type, n_context_dim, first_occurrence_only=False, mu_vec_np=None, alpha_mat_np=None,
                 lambda_mat_np=None, embedding_size=10, rnn_hidden_size=10, gap_mean=0, gap_scale=1):
        super().__init__(n_event_type, n_context_dim, first_occurrence_only, mu_vec_np, alpha_mat_np, lambda_mat_np)
        self.gap_mean = gap_mean
        self.gap_scale = gap_scale
        self.embeding = nn.Embedding(n_event_type, embedding_size)
        # + 1 for time gap
        self.rnn = nn.LSTM(embedding_size + 1, rnn_hidden_size)
        self.rnn_lin = nn.Linear(rnn_hidden_size, 1)
        self.graph_weights_to_current = None
        self.graph_weights_seq = None

    def _update_graph_weights(self):
        seq_type_event = self.seq_type_event
        seq_mask_to_current = self.seq_mask_to_current
        seq_time_to_end = self.seq_time_to_end

        # n_event_type: T * batch_size
        # event_embedding: T * batch_size * K
        event_embedding = self.embeding(seq_type_event)
        # time_gap: T * batch_size
        # (T - 1) * batch_size
        time_diff = (seq_time_to_end[:-1, :] - seq_time_to_end[1:, :] - self.gap_mean) / self.gap_scale
        init_time_diff = time_diff.new_zeros((1, time_diff.shape[1]))
        time_diff_input = torch.cat((init_time_diff, time_diff))
        rnn_input = torch.cat((event_embedding, time_diff_input[:, :, None]), dim=2)

        # rnn_output: T * batch_size * rnn_hidden_size
        rnn_output, _ = self.rnn(rnn_input)
        # graph_weights: T * batch_size
        # activation function
        graph_weights = torch.sigmoid(self.rnn_lin(rnn_output)).reshape(rnn_output.shape[:2])
        self.graph_weights_seq = graph_weights
        self.graph_weights_to_current = graph_weights[None, :, :] * seq_mask_to_current
