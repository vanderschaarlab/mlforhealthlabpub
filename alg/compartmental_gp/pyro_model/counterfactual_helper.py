import torch
import pyro_model.helper
import numpy as np
from pyro.ops.stats import quantile


def get_R0_sooner_lockdown(R0, lockdown_start_sooner):

    R0_last = R0[..., -1][:, :, None]
    R0_last_pad = R0_last.repeat(1, 1, lockdown_start_sooner)
    R0_counter = torch.cat([R0[..., lockdown_start_sooner:], R0_last_pad], dim=-1)
    return R0_counter

def get_R0_later_lockdown(R0, lockdown_start_later):

    R0_last = R0[..., 0][:, :, None]
    R0_last_pad = R0_last.repeat(1, 1, lockdown_start_later)
    R0_counter = torch.cat([R0_last_pad, R0[..., :-lockdown_start_later]], dim=-1)
    return R0_counter


def get_counterfactual(data_dict, forecaster, res, R0):
    p_fatal = res['p_fatal']
    case_import = res['case_import'] / forecaster.model.N

    infectious_days = res['infect_days']
    time_to_death = res['time_to_death']
    d_incubation = res['d_incubation']
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
    t_init = data_dict['t_init'].unsqueeze(0).repeat(i0.size(0), 1, 1)

    res_ode = pyro_model.helper.eluer_seir_time(s0, e0, i0, r0, f0, beta_t, sigma, alpha, p_fatal, d_death, case_import,
                                                t_init)
    r_t = res_ode[-1]
    prediction = r_t * forecaster.model.N
    # ..., t, p
    prediction = prediction.unsqueeze(-1).transpose(-1, -3)

    mask_half = res['mask_half']
    mask_full = torch.cat([mask_half, torch.zeros_like(mask_half)], dim=-1)[..., :-1]

    res_list = []
    for i in range(prediction.shape[0]):
        pred_temp = prediction[i, ...]
        mask_temp = mask_full[i, ...][0, ...].permute(1, 0, 2)
        prediction_conv = pred_temp.permute(2, 0, 1)

        res_inner_list = []

        for j in range(len(forecaster.model.N)):
            res_inner = torch.nn.functional.conv1d(prediction_conv[j:j + 1, ...], mask_temp[j:j + 1, ...],
                                                   padding=mask_half.shape[-1] - 1)
            res_inner_list.append(res_inner)

        res1 = torch.cat(res_inner_list, dim=0)
        res1 = res1.permute(1, 2, 0)
        res_list.append(res1)
    prediction = torch.stack(res_list, dim=0).numpy().squeeze()
    prediction = np.diff(prediction, axis=1)

    prediction = quantile(torch.tensor(prediction), (0.05, 0.5, 0.95), dim=0).numpy()
    return prediction