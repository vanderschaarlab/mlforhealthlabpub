import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.ops.stats import quantile

import forecast
import pyro_model.helper


def gumbel_softmax(logits, dim=-1, temperature=0.1, eps=1e-9):
    # get gumbel noise
    noise = torch.rand(logits.size(), dtype=logits.dtype, device=logits.device)
    noise = -1.0 * torch.log(noise + eps)
    noise = -1.0 * torch.log(noise + eps)

    x = (logits + noise) / temperature
    x = torch.softmax(x, dim=dim)
    return x

class CGP(forecast.ForecastingModel):

    def __init__(self, data_dict, dtype=torch.float, mask_size=14):
        super().__init__()
        self.n_country = len(data_dict['countries'])
        self.t_init = data_dict['t_init'].to(dtype)
        # d, 1
        self.N = torch.tensor(data_dict['population']).to(self.t_init).unsqueeze(-1).to(dtype)
        self.dtype = dtype
        self.mask_size = mask_size
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def model(self, zero_data, covariates):
        data_dim = zero_data.size(-1)
        time_dim = covariates.size(-2)

        # t, d, p
        # p[0] time, p[1] s_index
        covariates = pyro_model.helper.reshape_covariates_pyro(covariates, self.n_country).to(self.dtype)
        feature_dim = covariates.size(-1)
        time = covariates[..., 0:1]

        country_plate = pyro.plate("country", data_dim, dim=-2)
        mask_plate = pyro.plate("mask", self.mask_size, dim=-1)
        ######################### First GP: get time varying R0 #########################
        gp1_lengthscale = pyro.sample("r0_lengthscale", dist.LogNormal(np.log(7), 0.1))
        gp1_var = pyro.sample("r0_kernel_var", dist.LogNormal(0, 0.1))

        if gp1_lengthscale.dim() == 0:
            gp1_lengthscale = gp1_lengthscale.repeat(1, 1)
            gp1_var = gp1_var.repeat(1, 1)
        else:
            gp1_lengthscale = gp1_lengthscale.transpose(-1, -2)
            gp1_var = gp1_var.transpose(-1, -2)

        gp_covariates = covariates.transpose(1, 0)
        d_times_t = data_dim * time_dim
        # dxt, 1, p
        gp_covariates = gp_covariates.reshape(d_times_t, 1, gp_covariates.size(-1))

        var = pyro_model.helper.tensor_RBF(gp1_lengthscale, gp1_var, gp_covariates)
        var = var + torch.eye(var.shape[-1]) * 0.001
        assert var.size(-1) == d_times_t
        assert var.size(-2) == d_times_t
        assert var.size(-3) == 1

        # N, 1, dxt, dxt
        A = torch.cholesky(var)

        with country_plate:
            with self.time_plate:
                iid_n = pyro.sample("r0_iid_n", dist.Normal(0, 1))

        iid_n = iid_n.unsqueeze(-1)
        iid_n_shape = iid_n.dim()

        if iid_n_shape == 3:
            iid_n = iid_n.unsqueeze(0)
        # n, 1, dxt, 1
        iid_n = iid_n.reshape(iid_n.size(0), 1, d_times_t, 1)

        # n, 1, dxt, 1
        weight = torch.sigmoid(torch.einsum('abij,abjk->abik', A, iid_n))
        weight = weight[:, 0, ...]  # get rid of batch dimension
        weight = weight.reshape(weight.size(0), data_dim, time_dim, 1)
        if weight.size(0) == 1:
            weight = weight[0, ...]
        # (n), d, t

        # n, d, 1
        with country_plate:
            R00 = pyro.sample('R00', dist.Normal(2.65, 0.25))

        # n, d, t
        R0 = pyro.deterministic('R0', R00 * weight[..., 0])

        ######################### SIR: Solving ODE #########################
        with country_plate:
            p_fatal = pyro.sample('p_fatal', dist.Gamma(5., 5 / 2E-2))
            case_import = pyro.sample('case_import', dist.Normal(400, 100)) / self.N

        infectious_days = pyro.sample('infect_days', dist.LogNormal(np.log(2.5), 1.))
        time_to_death = pyro.sample('time_to_death', dist.Normal(11, 4.))
        d_incubation = pyro.sample('d_incubation', dist.LogNormal(np.log(2.6), 1.))
        alpha = 1. / d_incubation
        d_death = time_to_death - infectious_days

        e0 = case_import
        i0 = torch.zeros_like(e0)
        r0 = torch.zeros_like(e0)
        f0 = torch.zeros_like(e0)
        s0 = 1. - e0 - i0 - r0 - f0

        sigma = 1. / infectious_days
        beta_t = sigma * R0

        # t_init: same shape with i0
        t_init = self.t_init if i0.dim() == 2 else self.t_init.unsqueeze(0).repeat(i0.size(0), 1, 1)

        res_ode = pyro_model.helper.eluer_seir_time(s0, e0, i0, r0, f0, beta_t, sigma, alpha, p_fatal, d_death, case_import, t_init)
        r_t = res_ode[-1]
        prediction = r_t * self.N
        # ..., t, p
        prediction = prediction.unsqueeze(-1).transpose(-1, -3)
        prediction_no_lag = pyro.deterministic('prediction_no_lag', prediction)

        with country_plate:
            with mask_plate:
                lag_scores = pyro.sample('lag_score', dist.Normal(0., 1.))

        lag_logits = self.log_softmax(lag_scores)
        mask_half = pyro.deterministic('mask_half', gumbel_softmax(lag_logits))
        mask_full = torch.cat([mask_half, torch.zeros_like(mask_half)], dim=-1)[..., :-1]

        if prediction.dim() == 4:
            res_list = []
            for i in range(prediction.shape[0]):
                pred_temp = prediction[i, ...]
                mask_temp = mask_full[i, ...][:, None, :]
                prediction_conv = pred_temp.permute(2, 0, 1)

                res_inner_list = []

                for j in range(data_dim):
                    res_inner = torch.nn.functional.conv1d(prediction_conv[j:j + 1, ...], mask_temp[j:j + 1, ...],
                                                     padding=self.mask_size - 1)
                    res_inner_list.append(res_inner)

                res = torch.cat(res_inner_list, dim=0)
                res = res.permute(1, 2, 0)
                res_list.append(res)
            prediction = torch.stack(res_list, dim=0)
        else:
            # mask_full: 1, 1, M
            mask_full = mask_full[:, None, :]
            prediction_conv = prediction.permute(2, 0, 1)
            res_list = []
            for j in range(data_dim):
                res = torch.nn.functional.conv1d(prediction_conv[j:j+1, ...], mask_full[j:j+1, ...], padding=self.mask_size - 1)
                res_list.append(res)
            prediction = torch.cat(res_list, dim=0)
            prediction = prediction.permute(1, 2, 0)

        prediction = pyro.deterministic('prediction', prediction)

        assert prediction.shape[-2:] == zero_data.shape

        ######################### Second GP: output noise #########################
        # zero-mean GP noise
        with country_plate:
            gp2_lengthscale = pyro.sample('gp2_lengthscale', dist.Normal(14, 1))
            gp2_var = pyro.sample('gp2_var', dist.Normal(10, 1))

        time_x = torch.arange(time_dim) * 1.
        # t, d, 1
        time_x = time_x[:, None, None].repeat(1, data_dim, 1)
        time_var = pyro_model.helper.tensor_RBF(gp2_lengthscale, gp2_var, time_x)
        time_var = time_var + torch.eye(time_var.shape[-1]) * 0.01

        if time_var.shape[0] > 1:
            time_var = torch.mean(time_var, dim=0)
        else:
            time_var = time_var[0, ...]

        time_var = time_var.to(torch.double)
        mean_zero = torch.tensor(0)[None].to(torch.double)

        noise_dist = dist.MultivariateNormal(mean_zero, time_var)
        noise_dist = forecast.util.MVTNormalTime(noise_dist)

        self.predict(noise_dist, prediction)

    def get_R0(self, forecaster, Y_train, covariates_full, sample_size, batch_size):
        n_batch = sample_size // batch_size
        R_list = []
        map_list = []

        for i in range(n_batch):
            with torch.no_grad():
                with pyro.plate("particles", batch_size, dim=-3):
                    map_estimates = forecaster.guide(Y_train[:1, :], covariates_full)
                map_list.append(map_estimates)
                data_dim = Y_train.size(-1)
                time_dim = covariates_full.size(-2)

                kernel_lengthscale = map_estimates['r0_lengthscale']
                kernel_var = map_estimates['r0_kernel_var']
                covariates_pyro = pyro_model.helper.reshape_covariates_pyro(covariates_full,
                                                                            self.n_country).to(torch.float)

                kernel_lengthscale = kernel_lengthscale.transpose(-1, -2)
                kernel_var = kernel_var.transpose(-1, -2)

                # covariate
                covariates_pyro = covariates_pyro.transpose(1, 0)
                d_times_t = data_dim * time_dim
                # dxt, 1, p
                covariates_pyro = covariates_pyro.reshape(d_times_t, 1, covariates_pyro.size(-1))

                var = pyro_model.helper.tensor_RBF(kernel_lengthscale, kernel_var, covariates_pyro)
                var = var + torch.eye(var.shape[-1]) * 0.01
                A = torch.cholesky(var)

                iid_n = map_estimates['r0_iid_n']
                iid_n = torch.cat([iid_n, torch.randn(iid_n.size(0), iid_n.size(1), time_dim - iid_n.size(-1))], dim=-1)
                iid_n = iid_n.unsqueeze(-1)
                iid_n = iid_n.reshape(iid_n.size(0), 1, d_times_t, 1)

                weight = torch.sigmoid(torch.einsum('abij,abjk->abik', A, iid_n))
                weight = weight[:, 0, ...]  # get rid of batch dimension
                weight = weight.reshape(weight.size(0), data_dim, time_dim, 1)
                R00 = map_estimates['R00']
                R0 = R00 * weight[..., 0]
                R_list.append(R0)

        R0 = torch.cat(R_list, dim=0)
        R0low, R0mid, R0high = quantile(R0, (0.1, 0.5, 0.9), dim=0).squeeze(-1)

        map_estimates = dict.fromkeys(map_list[0].keys())
        for k in map_list[0].keys():
            k_list = []
            for m in map_list:
                k_list.append(m[k])
            kest = torch.cat(k_list, dim=0)
            map_estimates[k] = kest

        return R0low, R0mid, R0high, map_estimates


class ModelGPSEIRConvCountry(CGP):
    def __init__(self, data_dict, dtype=torch.float, mask_size=14):
        super().__init__(data_dict, dtype, mask_size)
