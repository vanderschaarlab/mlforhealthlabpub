# %% Necessary Packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import *


# %% GCIT Function
'''
Inputs:
 - z: Confounder variables, this is the conditioning set
 - x: First target variable
 - y: Second target variable

Hyper-parameters (=Default values)
 - mu: WGAN parameter = 1
 - eta: WGAN regularization parameter = 10
 - lamda: Information network parameter = 10
 - statistic: sets the comparison function between generated and true samples (rho in the algorithm pseudocode)

'''

def GCIT(x, y, z, statistic = "rdc", lamda = 10, normalize=True, verbose=False, n_iter=1000,debug=False):

    if normalize:
        z = (z - z.min()) / (z.max() - z.min())
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

    # %% Parameters
    # 1. # of samples
    n = len(z[:, 0])

    # define training and testing subsets, training for learning the sampler and 
    # testing for computing test statistic. Set 2/3 and 1/3 as default
    x_train, y_train, z_train = x[:int(2*n/3),], y[:int(2*n/3),], z[:int(2*n/3),] 
    x_test, y_test, z_test = x[int(2*n/3):,], y[int(2*n/3):,], z[int(2*n/3):,] 

    n = len(z_train[:, 0])
    
    # 2. # of confounders
    z_dim = len(z_train[0, :])

    # 3. # of target variables of interest (can be changed if needed)
    x_dim = 1

    # 3. # of random and hidden dimensions
    if z_dim <= 20:
        v_dim = int(3)
        h_dim = int(3)

    else:
        v_dim = int(z_dim / 10)
        h_dim = int(z_dim / 10)

    # 4. size of minibatch
    mb_size = 64

    # 5. WGAN parameters
    eta = 10
    lr = 1e-4

    # %% Necessary Functions

    # 1. Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    # 2. Sample from normal distribution: Random variable generation
    def sample_V(m, n):
        #out = np.random.rand(m, n)
        #out = np.random.laplace(loc=0.0, scale=1.0, size=n * m)
        #out = np.reshape(out, (m, n))
        out = np.random.normal(0., np.sqrt(1. / 3), size=[m, n])
        return out

    # 3. Sample from the real data (Mini-batch index sampling)
    def sample_Z(m, n):
        return np.random.permutation(m)[:n]

    # 4. Permutation for MINE computation
    def Permute(x):
        n = len(x)
        idx = np.random.permutation(n)
        out = x[idx]
        return out


    # %% Placeholders

    # 1. Target Feature
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_dim])
    # 2. Target Permuted Feature
    X_hat = tf.compat.v1.placeholder(tf.float32, shape=[None, x_dim])
    # 3. Random Variable V
    V = tf.compat.v1.placeholder(tf.float32, shape=[None, v_dim])
    # 3. Confounder Z
    Z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])

    # %% Network Building

    # %% 1. WGAN Discriminator 
    # - one hidden layer as default even though more may be needed for complex data generation processes
    WD_W1 = tf.Variable(xavier_init([x_dim + z_dim, h_dim]))
    WD_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    WD_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    WD_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    WD_W3 = tf.Variable(xavier_init([h_dim, x_dim]))
    WD_b3 = tf.Variable(tf.zeros(shape=[x_dim]))

    theta_WD = [WD_W1, WD_W3, WD_b1, WD_b3]

    # %% 2. Generator
    # Input: Z and V
    G_W1 = tf.Variable(xavier_init([z_dim + v_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, x_dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[x_dim]))

    theta_G = [G_W1, G_W3, G_b1, G_b3]

    # %% 3. MINE
    # Input: X and tilde X
    # For X
    M_W1A = tf.Variable(xavier_init([x_dim]))
    M_W1B = tf.Variable(xavier_init([x_dim]))
    M_b1 = tf.Variable(tf.zeros(shape=[x_dim]))

    # For tilde X
    M_W2A = tf.Variable(xavier_init([x_dim]))
    M_W2B = tf.Variable(xavier_init([x_dim]))
    M_b2 = tf.Variable(tf.zeros(shape=[x_dim]))

    # Combine
    M_W3 = tf.Variable(xavier_init([x_dim]))
    M_b3 = tf.Variable(tf.zeros(shape=[x_dim]))

    theta_M = [M_W1A, M_W1B, M_W2A, M_W2B, M_W3, M_b1, M_b2, M_b3]

    # %% Functions
    # 1. Generator
    def generator(z, v):
        inputs = tf.concat(axis=1, values=[z, v])
        G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
        #G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        #G_out = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)

        G_out = tf.nn.sigmoid(tf.matmul(G_h1, G_W3) + G_b3)

        return G_out

    # 2. WGAN Discriminator
    def WGAN_discriminator(x, z):
        inputs = tf.concat(axis=1, values=[x, z])
        WD_h1 = tf.nn.relu(tf.matmul(inputs, WD_W1) + WD_b1)
        #WD_h2 = tf.nn.relu(tf.matmul(WD_h1, WD_W2) + WD_b2)
        #WD_out = (tf.matmul(WD_h2, WD_W3) + WD_b3)

        WD_out = (tf.matmul(WD_h1, WD_W3) + WD_b3)

        return WD_out

    # 3. MINE
    def MINE(x, x_hat):
        M_h1 = tf.nn.tanh(M_W1A * x + M_W1B * x_hat + M_b1)
        M_h2 = tf.nn.tanh(M_W2A * x + M_W2B * x_hat + M_b2)
        M_out = (M_W3 * (M_h1 + M_h2) + M_b3)

        Exp_M_out = tf.exp(M_out)

        return M_out, Exp_M_out

    # %% Combination across the networks
    # 1. Generator
    G_sample = generator(Z, V)

    # 2. WGAN Outputs for real and fake
    WD_real = WGAN_discriminator(X, Z)
    WD_fake = WGAN_discriminator(G_sample, Z)

    # 3. MINE Computation
    # Without permutation
    M_out, _ = MINE(X, G_sample)
    # With permutation
    _, Exp_M_out = MINE(X_hat, G_sample)

    # 4. WGAN Loss Replacement of Clipping algorithm to Penalty term
    eps = tf.random.uniform([mb_size, 1], minval=0., maxval=1.)
    X_inter = eps * X + (1. - eps) * G_sample

    grad = tf.gradients(WGAN_discriminator(X_inter, Z), [X_inter])[0]
    grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2 + 1e-8, axis=1))
    grad_pen = eta * tf.reduce_mean((grad_norm - 1) ** 2)

    # %% Loss function
    # 1. WGAN Loss, aim to make WD_fake small and WD_real big
    WD_loss = tf.reduce_mean(WD_fake) - tf.reduce_mean(WD_real) + grad_pen

    # 2. MINE Loss
    M_loss = lamda * (tf.reduce_sum(tf.reduce_mean(M_out, axis=0) - \
                                    tf.math.log(tf.reduce_mean(Exp_M_out, axis=0))))

    # 3. Generator loss, aim make WD_fake high
    G_loss = -tf.reduce_mean(WD_fake) + lamda * M_loss

    # Solver
    WD_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(WD_loss, \
                                                                                        var_list=theta_WD))
    G_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=theta_G))
    M_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(-M_loss, var_list=theta_M))

    # %% Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    Generator_loss = []
    Mine_loss = []
    WDiscriminator_loss = []

    # %% Iterations
    for it in range(n_iter):

        for _ in range(5):
            # %% WGAN and MINE Training

            # Random variable generation
            V_mb = sample_V(mb_size, v_dim)

            # Minibatch sampling
            Z_idx = sample_Z(n, mb_size)
            Z_mb = z_train[Z_idx, :]
            X_mb = x_train[Z_idx]
            X_perm_mb = Permute(X_mb)

            # 1. WGAN Training
            _, WD_loss_curr = sess.run([WD_solver, WD_loss], \
                                       feed_dict={X: X_mb, Z: Z_mb, V: V_mb, X_hat: X_perm_mb})
            # 2. MINE Training
            _, M_loss_curr = sess.run([M_solver, M_loss], \
                                      feed_dict={X: X_mb, Z: Z_mb, V: V_mb, X_hat: X_perm_mb})

        # Random variable generation
        V_mb = sample_V(mb_size, v_dim)

        # Minibatch sampling
        Z_idx = sample_Z(n, mb_size)
        Z_mb = z_train[Z_idx, :]
        X_mb = x_train[Z_idx]
        X_perm_mb = Permute(X_mb)

        # Generator training
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, Z: Z_mb, V: V_mb, X_hat: X_perm_mb})

        Generator_loss.append(G_loss_curr)
        WDiscriminator_loss.append(WD_loss_curr)
        Mine_loss.append(M_loss_curr)

        # %% Intermediate Losses
        if verbose and it % 499 == 0:
            print('Iter: {}'.format(it))
            print('Generator_loss: {:.4}'.format(G_loss_curr))
            print('WD_loss: {:.4}'.format(WD_loss_curr))
            print('M_loss: {:.4}'.format(M_loss_curr))
            print()
            
        # stop training if discriminator loss is sufficiently low (heuristic)
        if abs(WD_loss_curr) < 0.1:
            break
            
    if verbose:
        # plot training losses
        plt.plot(range(n_iter), WDiscriminator_loss, range(n_iter), Generator_loss, range(n_iter), Mine_loss)
        plt.legend(('WGAN Discriminator', 'Generator', 'MINE'),
                   loc='upper right')
        plt.title('Training losses')
        plt.tight_layout()
        plt.show()

    # %% Compute test statistic
    # 1. Number of samples for null computation
    n_samples = 1000
    rho = []

    # 2. Choice of statistic
    if statistic == "corr":
        stat = correlation
    if statistic == "mmd":
        stat = mmd_squared
    if statistic == "kolmogorov":
        stat = kolmogorov
    if statistic == "wilcox":
        stat = wilcox
    if statistic == "rdc":
        stat = rdc
   
    # 3. Generate and collect samples on testing data
    for sample in range(n_samples):
        x_hat = sess.run([G_sample], feed_dict={Z: z_test, V: sample_V(len(z_test[:, 0]), v_dim)})[0]
        rho.append(stat(x_hat, y_test))

    # 4. p-value computation as a two-sided test
    p_value = min(sum(rho < stat(x_test.reshape(len(x_test)), y_test)) / n_samples,
                  sum(rho > stat(x_test.reshape(len(x_test)), y_test)) / n_samples)

    
    if debug:
        print('Statistics of x_hat ', stats.describe(x_hat))
        print('Statistics of x_train ',stats.describe(x_test))
        print('Statistics of generated rho ', stats.describe(rho))
        print('Observed rho',stat(x_test.reshape(len(x_test)), y_test))

    return(p_value)
