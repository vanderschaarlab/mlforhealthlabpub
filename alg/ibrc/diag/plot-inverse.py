import dill
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

plt.figure(figsize=(6,3))

with open('diag/res/alp-lowlow.obj', 'rb') as f:
    alphas1 = dill.load(f)[:,0]
with open('diag/res/alp-med.obj', 'rb') as f:
    alphas2 = dill.load(f)[:,0]
with open('diag/res/alp-hgh.obj', 'rb') as f:
    alphas3 = dill.load(f)[:,0]

pdf1 = stats.gaussian_kde(alphas1)
pdf2 = stats.gaussian_kde(alphas2)
pdf3 = stats.gaussian_kde(alphas3)

x1s = np.linspace(0, .001, 1000)
x2s = np.linspace(.4, .6, 1000)
x3s = np.linspace(9, 11, 1000)

plt.subplot(3, 1, 1)
plt.plot(x1s, pdf1(x1s), color='tab:blue', label=r'$\alpha^{\mathrm{true}}\,\to\,0^{\!+}$')
plt.axvline(x=x1s[pdf1(x1s).argmax()], color='tab:blue', alpha=.5)
plt.legend()
plt.yticks([])

plt.subplot(3, 1, 2)
plt.plot(x2s, pdf2(x2s), color='tab:orange', label=r'$\alpha^{\mathrm{true}}=0.5$')
plt.axvline(x=x2s[pdf2(x2s).argmax()], color='tab:orange', alpha=.5)
plt.axvline(x=.5, color='gray', alpha=.5)
plt.yticks([])
plt.legend()
plt.yticks([])
plt.ylabel(r'$\mathbb{P}(\alpha|\mathcal{D})$')


plt.subplot(3, 1, 3)
plt.plot(x3s, pdf3(x3s), color='tab:green', label=r'$\alpha^{\mathrm{true}}=10$')
plt.axvline(x=x3s[pdf3(x3s).argmax()], color='tab:green', alpha=.5)
plt.axvline(x=10, color='gray', alpha=.5)
plt.legend()
plt.yticks([])
plt.xlabel(r'$\alpha$')

plt.tight_layout(pad=0)
plt.savefig('diag/fig/inverse-a.pdf')

###

fig = plt.figure(figsize=(6,3))

with open('diag/res/eta.obj', 'rb') as f:
    betaetas = dill.load(f)[:,1:]
pdf = stats.gaussian_kde(betaetas.T)

xs = np.linspace(0, 200000, 100)
ys = np.linspace(50, 100, 100)
xs, ys = np.meshgrid(xs, ys)
zs = pdf(np.stack((xs.ravel(), ys.ravel()))).reshape(100, 100)

i_max = pdf(np.stack((xs.ravel(), ys.ravel()))).argmax()
x_max = xs.ravel()[i_max]
y_max = ys.ravel()[i_max]
z_max = zs.ravel()[i_max]

ax = fig.gca(projection='3d')
ax.plot_trisurf(xs.ravel(), ys.ravel(), zs.ravel(), cmap=cm.coolwarm, alpha=.75, linewidth=0)
ax.plot([1e3,1e3], [75,75], [0,z_max], color='black')
ax.plot([x_max,x_max], [y_max,y_max], [0,z_max], color='red')

ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\mathbb{P}(\beta,\eta|\mathcal{D})$')
ax.set_zticklabels([])


ax.view_init(15, -60)
ax.invert_xaxis()
ax.set_box_aspect((1, 1, .4), zoom=1.2)

fig.tight_layout(pad=0)
fig.subplots_adjust(top=1.15,bottom=-0.05)
plt.savefig('diag/fig/inverse-b.pdf')

###

fig = plt.figure(figsize=(6,3))

with open('diag/res/bet.obj', 'rb') as f:
    betaetas = dill.load(f)[:,1:]
pdf = stats.gaussian_kde(betaetas.T)

xs = np.linspace(1, 1.5, 100)
ys = np.linspace(0, 4, 100)
xs, ys = np.meshgrid(xs, ys)
zs = pdf(np.stack((xs.ravel(), ys.ravel()))).reshape(100, 100)

i_max = pdf(np.stack((xs.ravel(), ys.ravel()))).argmax()
x_max = xs.ravel()[i_max]
y_max = ys.ravel()[i_max]
z_max = zs.ravel()[i_max]

ax = fig.gca(projection='3d')
ax.plot_trisurf(xs.ravel(), ys.ravel(), zs.ravel(), cmap=cm.coolwarm, alpha=.75, linewidth=0)
ax.plot([1.25,1.25], [1e-3,1e-3], [0,z_max], color='black')
ax.plot([x_max,x_max], [y_max,y_max], [0,z_max], color='red')

ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\eta$')
ax.set_zlabel(r'$\mathbb{P}(\beta,\eta|\mathcal{D})$')
ax.set_zticklabels([])

ax.view_init(15, -60)
ax.set_box_aspect((1, 1, .4), zoom=1.2)

fig.tight_layout(pad=0)
fig.subplots_adjust(top=1.15,bottom=-0.05)
plt.savefig('diag/fig/inverse-c.pdf')
