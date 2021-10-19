import jax
import jax.numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True

from constants import *
from functions import *

inf = 1e3
eps = 1e-3

def add_plot(alpha, beta, eta):
    hyper = (gamma, alpha, beta, eta)
    *_, pi, xi = solve(tilde_pi, tilde_xi, upsilon, hyper, 100)

    zs = list()
    z = ext_z0.copy()
    zs.append(z[1])
    for _ in range(3):
        z = rho(z, 2, 1, xi)
        zs.append(z[1])

    ls = plt.plot(val_z[:,1], pi)
    l = plt.plot(zs, [-0.1] * 4, marker='^', linestyle='none')

    plt.ylim(-0.2, 1.1)
    return ls, l

plt.figure(figsize=(6,2))
add_plot(eps, inf, eps)
plt.xlabel('$p(s_{+}|z)$')
plt.legend([r'$\pi(u_{-}|z)$', r'$\pi(u_{+}|z)$', r'$\pi(u_{=}|z)$',
    r'$\{z_t\}:z_{t+1}\sim\rho_{\sigma}(\cdot|z_t,u_{=},x_{+})$'], loc='upper center')
plt.tight_layout(pad=0)
plt.savefig('diag/fig/forward-a.pdf')

plt.figure(figsize=(6,2))
add_plot(.5, 1.25, eps)
plt.xlabel('$p(s_{+}|z)$')
plt.tight_layout(pad=0)
plt.savefig('diag/fig/forward-b.pdf')

plt.figure(figsize=(6,2))
add_plot(.5, inf, eps)
plt.xlabel('$p(s_{+}|z)$')
plt.tight_layout(pad=0)
plt.savefig('diag/fig/forward-c.pdf')

plt.figure(figsize=(6,2))
add_plot(10, inf, eps)
plt.xlabel('$p(s_{+}|z)$')
plt.tight_layout(pad=0)
plt.savefig('diag/fig/forward-a1.pdf')

plt.figure(figsize=(6,2))
add_plot(.5, -.75, eps)
plt.xlabel('$p(s_{+}|z)$')
plt.tight_layout(pad=0)
plt.savefig('diag/fig/forward-b1.pdf')

plt.figure(figsize=(6,2))
add_plot(.5, inf, 75)
plt.xlabel('$p(s_{+}|z)$')
plt.tight_layout(pad=0)
plt.savefig('diag/fig/forward-c1.pdf')
