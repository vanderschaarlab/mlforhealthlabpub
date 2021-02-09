import torch
import numpy as np


def tensor_scaled_dist(l, x):
    # l: d, 1 or N, d, 1 => N, d, 1
    if l.dim() == 2:
        l = l.unsqueeze(0)

    # N, d, 1 => N, 1, 1, d
    l = l.unsqueeze(1).transpose(-1, -2)

    # t, d, p => t, p, d
    x = x.transpose(-1, -2)

    # N, t, p, d => N, d, t, p
    x_scaled = (x / l).permute((0, 3, 1, 2))

    # N, d, t, 1
    x2 = (x_scaled ** 2).sum(-1, keepdim=True)

    # N, d, t, t
    xz = torch.einsum('abij,abjk->abik', x_scaled, x_scaled.transpose(-1, -2))

    # N, d, t, t
    r2 = x2 - 2 * xz + x2.transpose(-1, -2)
    return r2.clamp(min=0)


def tensor_RBF(l, v, x):
    # N, d, t, t
    r2 = torch.exp(-0.5 * tensor_scaled_dist(l, x))
    # d, t, t, N
    r2 = r2.permute((1, 2, 3, 0))

    # v: d, 1 or N, d, 1 => N, d, 1
    if v.dim() == 2:
        v = v.unsqueeze(0)

    # N, d, 1, 1
    v = v.unsqueeze(-1)
    # d, 1, 1, N
    v = v.permute((1, 2, 3, 0))

    res = (v * r2).permute(3, 0, 1, 2)

    return res


def eluer_seir_time(s0, e0, i0, r0, f0, beta_t, sigma, alpha, p_fatal, D_death, case_import, t_init):
    # i0, r0, sigma: D, 1; N, D, 1
    # beta_t: D, T; N, D, T
    # t_init: make sure same shape as i0 and r0 (reshape if necessary)

    # i0 = torch.zeros(500, 2, 1)
    # r0 = torch.zeros(500, 2, 1)
    # e0 = torch.zeros(500, 2, 1) + 500 / N
    # f0 = torch.zeros(500, 2, 1)
    # s0 = 1 - i0 - r0 - e0 - f0
    #
    # #  alpha, p_fatal, D_death, case_import
    #
    # # beta_t = torch.randn(500, 2, 37) * 0.001 + 0.5
    # sigma = torch.zeros(500, 2, 1) + 1. / D_infectious
    # alpha = torch.zeros(500, 2, 1) + 1. / D_incubation
    # p_fatal = torch.zeros(500, 2, 1) + 0.012
    # D_death = torch.zeros(500, 2, 1) + Time_to_death - D_infectious
    # case_import = torch.zeros(500, 2, 1) + 500 / N
    # beta_t = (sigma * uk_r0).to(case_import)
    #
    # t_init = torch.zeros((2, 1))
    # t_init[0, 0] = 0
    # t_init[1, 0] = 5
    #
    # t_init = t_init.unsqueeze(0).repeat(500, 1, 1)
    # s, e, i, r, f = eluer_seir_time(s0, e0, i0, r0, f0, beta_t, sigma, alpha, p_fatal, D_death, case_import, t_init)

    T = beta_t.size(-1)

    s_list = []
    e_list = []
    i_list = []
    r_list = []
    f_list = []

    s_t = torch.zeros_like(s0, dtype=torch.float)
    e_t = torch.zeros_like(e0, dtype=torch.float)
    i_t = torch.zeros_like(i0, dtype=torch.float)
    r_t = torch.zeros_like(r0, dtype=torch.float)
    f_t = torch.zeros_like(f0, dtype=torch.float)

    for t in range(T):
        i_t[t_init > t] = 0.
        r_t[t_init > t] = 0.
        s_t[t_init > t] = 0.
        e_t[t_init > t] = 0.
        f_t[t_init > t] = 0.

        i_t[t_init == t] = i0[t_init == t]
        r_t[t_init == t] = r0[t_init == t]
        s_t[t_init == t] = s0[t_init == t]
        e_t[t_init == t] = e0[t_init == t]
        f_t[t_init == t] = f0[t_init == t]

        i_list.append(i_t)
        r_list.append(r_t)
        s_list.append(s_t)
        e_list.append(e_t)
        f_list.append(f_t)

        dSdt = -beta_t[..., t:t + 1] * s_t * i_t
        dEdt = beta_t[..., t:t + 1] * s_t * i_t - alpha * e_t + case_import
        dIdt = alpha * e_t - sigma * i_t
        dRdt = p_fatal * sigma * i_t - (1 / D_death) * r_t
        dFdt = (1 / D_death) * r_t

        i_t = i_t + dIdt
        r_t = r_t + dRdt
        s_t = s_t + dSdt
        e_t = e_t + dEdt
        f_t = f_t + dFdt

    i = torch.cat(i_list, dim=-1)
    r = torch.cat(r_list, dim=-1)
    s = torch.cat(s_list, dim=-1)
    e = torch.cat(e_list, dim=-1)
    f = torch.cat(f_list, dim=-1)
    return s, e, i, r, f


def eluer_sir_time(i0, r0, beta_t, sigma, t_init):
    # i0, r0, sigma: D, 1; N, D, 1
    # beta_t: D, T; N, D, T
    # t_init: make sure same shape as i0 and r0 (reshape if necessary)

    # test 1
    # i0 = torch.randn(500, 2, 1) * 0.001 + 0.01
    # r0 = torch.randn(500, 2, 1) * 0.001 + 0.01
    # beta_t = torch.randn(500, 2, 37) * 0.001 + 0.5
    # sigma = torch.randn(500, 2, 1) * 0.001 + 0.01
    #
    # t_init = torch.zeros((2, 1))
    # t_init[0, 0] = 0
    # t_init[1, 0] = 5
    #
    # t_init = t_init.unsqueeze(0).repeat(500, 1, 1)

    # test 2
    # i0 = torch.randn(2, 1) * 0.001 + 0.01
    # r0 = torch.randn(2, 1) * 0.001 + 0.01
    # beta_t = torch.randn(2, 37) * 0.001 + 0.5
    # sigma = torch.randn(2, 1) * 0.001 + 0.01
    #
    # t_init = torch.zeros_like(i0)
    # t_init[0, 0] = 0
    # t_init[1, 0] = 5

    # i, r = eluer_sir(i0, r0, beta_t, sigma, t_init)

    T = beta_t.size(-1)

    i_list = []
    r_list = []

    #     i_t = i0.clone()
    #     r_t = r0.clone()

    i_t = torch.zeros_like(i0, dtype=torch.float)
    r_t = torch.zeros_like(r0, dtype=torch.float)

    for t in range(T):
        i_t[t_init > t] = 0.
        r_t[t_init > t] = 0.

        i_t[t_init == t] = i0[t_init == t]
        r_t[t_init == t] = r0[t_init == t]

        i_list.append(i_t)
        r_list.append(r_t)

        delta_1 = beta_t[..., t:t + 1] * i_t * (1. - r_t)
        delta_2 = sigma * i_t

        i_t = i_t + delta_1 - delta_2
        r_t = r_t + delta_1

    i = torch.cat(i_list, dim=-1)
    r = torch.cat(r_list, dim=-1)
    return i, r


def block_diag(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args )

    shapes = torch.tensor([a.shape for a in arrs])
    out = torch.zeros(torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def get_covariates(data_dict, train_len, notime=False):
    n_country = len(data_dict['countries'])
    s_ind_train = data_dict['s_index'][:train_len, ...]
    country_feat = data_dict['country_feat'][None, :, :]
    country_feat = country_feat.repeat(train_len, 1, 1)

    if not notime:
        time_feat = torch.arange(train_len).to(s_ind_train).unsqueeze(-1).repeat(1, n_country)
        covariate_stack = torch.stack([time_feat, s_ind_train], dim=-1)
    else:
        covariate_stack = s_ind_train[..., None]
    new_shape = covariate_stack.size(-1) * covariate_stack.size(-2)
    covariate = covariate_stack.view(covariate_stack.size(0), new_shape)
    return covariate


def reshape_covariates_pyro(covariate, n_country):
    p_total = covariate.size(-1)
    covariate_unstack = covariate.view(covariate.size(0), n_country, p_total//n_country)
    return covariate_unstack


def get_Y(data_dict, train_len, daily=False):
    if not daily:
        return data_dict['cum_death'][:train_len]
    else:
        return data_dict['daily_death'][:train_len]

def get_covariates_intervention(data_dict, train_len, notime=False):
    n_country = len(data_dict['countries'])
    i_ind_train = data_dict['i_index'][:train_len, ...]
    country_feat = data_dict['country_feat'][None, :, :]
    country_feat = country_feat.repeat(train_len, 1, 1)

    if not notime:
        time_feat = torch.arange(train_len).to(i_ind_train)[:, None, None].repeat(1, n_country, 1) / 100
        covariate_stack = torch.cat([time_feat, i_ind_train, country_feat], dim=-1)
    else:
        covariate_stack = torch.cat([i_ind_train, country_feat], dim=-1)
    new_shape = covariate_stack.size(-1) * covariate_stack.size(-2)
    covariate = covariate_stack.view(covariate_stack.size(0), new_shape)
    return covariate


def smooth_curve_1d(x):
    w = np.ones(7, 'd')
    y = np.convolve(w / w.sum(), x, mode='valid')
    y = np.concatenate([np.zeros(3), y])
    return y


def smooth_daily(data_dict):
    daily = data_dict['daily_death']

    dy_list = list()
    for i in range(daily.size(1)):
        ds = daily[:, i]
        dy = smooth_curve_1d(ds)
        dy_list.append(dy)

    sy = np.stack(dy_list, axis=-1)
    cum_y = np.cumsum(sy, axis=0)
    new_len = min(cum_y.shape[0], data_dict['i_index'].shape[0])

    return {
        'cum_death': torch.tensor(cum_y)[:new_len, :],
        'daily_death': torch.tensor(sy)[:new_len, :],
        'actual_daily_death': data_dict['daily_death'][:new_len, :],
        'actual_cum_death': data_dict['cum_death'][:new_len, :],
        's_index': data_dict['s_index'][:new_len, :],
        'i_index': data_dict['i_index'][:new_len, :],
        'population': data_dict['population'],
        't_init': data_dict['t_init'],
        'date_list': data_dict['date_list'][:new_len],
        'countries': data_dict['countries'],
        'country_feat': data_dict['country_feat']
    }

