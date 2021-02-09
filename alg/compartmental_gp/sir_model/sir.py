import numpy as np
from scipy.integrate import odeint
import torch


def sigmoid(x):
    return 1. / (1. + np.exp(-1. * x))


def softplus(x):
    return np.log(1 + np.exp(x)) if x < 10 else x


class SIR:
    def __init__(self, beta=0.15, gamma=1. / 20):
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        self.beta, self.gamma = beta, gamma

    @staticmethod
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I

        return dSdt, dIdt, dRdt

    def solve(self, t, *init_cond):
        S0, I0, R0 = init_cond
        N = S0 + I0 + R0

        # Integrate the SIR equations over the time grid, t.
        ret = odeint(SIR.deriv, init_cond, t, args=(N, self.beta, self.gamma))
        S, I, R = ret.T
        return S, I, R


class SEIR:
    def __init__(self, beta=0.15, gamma=1. / 20, alpha=1. / 5):
        # alpha: 1 / incubation period
        self.beta, self.gamma, self.alpha = beta, gamma, alpha

    @staticmethod
    def deriv(y, t, N, beta, gamma, alpha):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * I * S / N - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    @staticmethod
    def deriv_mcmc(y, t, p):
        N = p[0]
        beta = p[1]
        gamma = p[2]
        alpha = p[3]

        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]
        dSdt = -beta * S * I / N
        dEdt = beta * I * S / N - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def solve(self, t, *init_cond):
        S0, E0, I0, R0 = init_cond
        N = S0 + E0 + I0 + R0

        # Integrate the SIR equations over the time grid, t.
        ret = odeint(SEIR.deriv, init_cond, t, args=(N, self.beta, self.gamma, self.alpha))
        S, E, I, R = ret.T
        return S, E, I, R

def oxford_sir(y, t, p):
    beta = p[0]
    sigma = p[1]

    # y: infected
    # z: removed
    z = y[1]
    y = y[0]

    dydt = beta * y * (1 - z) - sigma * y
    dzdt = beta * y * (1 - z)

    return [dydt, dzdt]


def f_exp(x, alpha, beta):
    return alpha * np.power(2, x * beta)


def f_factory_opt(N, t_start, t_end):
    def f(x, R0, infectious_days, Psi, theta):
        i0 = 1. / N
        r0 = 0.

        # make positive
        # infectious_days = softplus(infectious_days)
        infectious_days = 4.5

        sigma = 1. / infectious_days
        # breaks optimization
        Psi = np.int(Psi)

        beta = sigma * R0

        t = np.arange(t_end + Psi)

        # solve ODE

        ret = odeint(oxford_sir, [i0, r0], t, args=((beta, sigma),))
        sir_curves = ret.T

        z_t = sir_curves[1]

        Y_t = theta * z_t[(t_start + Psi): (t_end + Psi)] * N

        return Y_t

    return f


def intervention_sir(y, t, p):
    beta = p[0]
    sigma = p[1]

    # y: infected
    # z: removed
    z = y[1]
    y = y[0]

    beta_t = np.interp(t, np.arange(len(beta)), beta)

    dydt = beta_t * y * (1 - z) - sigma * y
    dzdt = beta_t * y * (1 - z)

    return [dydt, dzdt]


def f_intervention_factory(N, t_start, t_end, s_index, s_ind, infectious_days=4.5, Psi=30):
    def f(x, R0, theta, s_b, s_a):
        i0 = 1. / N
        r0 = 0.

        sigma = 1. / infectious_days

        t = np.arange(t_end + Psi)

        # beta depends on s_index
        s = s_index[s_ind + t_start - Psi: s_ind + t_start + t_end]
        beta = sigma * R0 * 2 * sigmoid(-1. * s_b * s + s_a)

        # solve ODE

        ret = odeint(intervention_sir, [i0, r0], t, args=((beta, sigma),))
        sir_curves = ret.T

        z_t = sir_curves[1]

        Y_t = theta * z_t[(t_start + Psi): (t_end + Psi)] * N

        return Y_t

    return f


def eluer_sir(i0, r0, beta_t, sigma):
    T = beta_t.size(-1)

    i_list = []
    r_list = []

    i_t = i0
    r_t = r0

    for t in range(T):
        i_list.append(i_t)
        r_list.append(r_t)

        delta_1 = beta_t[..., t:t + 1] * i_t * (1. - r_t)
        delta_2 = sigma * i_t

        i_t = i_t + delta_1 - delta_2
        r_t = r_t + delta_1

    i = torch.cat(i_list, dim=-1)
    r = torch.cat(r_list, dim=-1)
    return i, r


if __name__ == '__main__':
    sir = SIR()
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)

    e0, i0, r0 = 0, 1, 0
    n = 50000000
    s0 = n - i0 - r0 - e0

    res = sir.solve(t, s0, i0, r0)
    print(res[0][:10])
    print(res[1][:10])
    print(res[2][:10])

    seir = SEIR()
    res = seir.solve(t, s0, e0, i0, r0)
    print(res[0][:10])
    print(res[1][:10])
    print(res[2][:10])
    print(res[3][:10])

