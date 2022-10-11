'''
Written by Jinsung Yoon
Date: Jul 9th 2018 (Revised Oct 19th 2018)
Generative Adversarial Imputation Networks (GAIN) Implementation on MNIST
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf
Appendix Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf
Contact: jsyoon0823@g.ucla.edu
'''

#%% Packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm

#%% System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.5
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Imput Dim (Fixed)
Dim = 784
# 6. No
Train_No = 55000
Test_No = 10000

#%% Data Input
# MNIST
mnist = input_data.read_data_sets('../../MNIST_data', one_hot = True)

# X
trainX, _ = mnist.train.next_batch(Train_No) 
testX, _  = mnist.test.next_batch(Test_No) 

# Mask Vector and Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C
  
trainM = sample_M(Train_No, Dim, p_miss)
testM = sample_M(Test_No, Dim, p_miss)

#%% Necessary Functions
# 1. Xavier Initialization Definition
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)
    
# 2. Plot (4 x 4 subfigures)
def plot(samples):
    fig = plt.figure(figsize = (5,5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        
    return fig
   
'''
GAIN Consists of 3 Components
- Generator
- Discriminator
- Hint Mechanism
'''   
   
#%% GAIN Architecture   
   
#%% 1. Input Placeholders
# 1.1. Data Vector
X = tf.placeholder(tf.float32, shape = [None, Dim])
# 1.2. Mask Vector 
M = tf.placeholder(tf.float32, shape = [None, Dim])
# 1.3. Hint vector
H = tf.placeholder(tf.float32, shape = [None, Dim])
# 1.4. Random Noise Vector
Z = tf.placeholder(tf.float32, shape = [None, Dim])

#%% 2. Discriminator
D_W1 = tf.Variable(xavier_init([Dim*2, 256]))     # Data + Hint as inputs
D_b1 = tf.Variable(tf.zeros(shape = [256]))

D_W2 = tf.Variable(xavier_init([256, 128]))
D_b2 = tf.Variable(tf.zeros(shape = [128]))

D_W3 = tf.Variable(xavier_init([128, Dim]))
D_b3 = tf.Variable(tf.zeros(shape = [Dim]))       # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

#%% 3. Generator
G_W1 = tf.Variable(xavier_init([Dim*2, 256]))     # Data + Mask as inputs (Random Noises are in Missing Components)
G_b1 = tf.Variable(tf.zeros(shape = [256]))

G_W2 = tf.Variable(xavier_init([256, 128]))
G_b2 = tf.Variable(tf.zeros(shape = [128]))

G_W3 = tf.Variable(xavier_init([128, Dim]))
G_b3 = tf.Variable(tf.zeros(shape = [Dim]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

#%% GAIN Function

#%% 1. Generator
def generator(x,z,m):
    inp = m * x + (1-m) * z  # Fill in random noise on the missing values
    inputs = tf.concat(axis = 1, values = [inp,m])  # Mask + Data Concatenate
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
    
    return G_prob
    
#%% 2. Discriminator
def discriminator(x, m, g, h):
    inp = m * x + (1-m) * g  # Replace missing values to the imputed values
    inputs = tf.concat(axis = 1, values = [inp,h])  # Hint + Data Concatenate
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
    
    return D_prob

#%% 3. Others
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 1., size = [m, n])        

def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

#%% Structure
G_sample = generator(X,Z,M)
D_prob = discriminator(X, M, G_sample, H)

#%% Loss
D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8)) * 2
G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8)) / tf.reduce_mean(1-M)
MSE_train_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

D_loss = D_loss1
G_loss = G_loss1  + alpha * MSE_train_loss 

#%% MSE Performance metric
MSE_test_loss = tf.reduce_mean(((1-M) * X - (1-M)*G_sample)**2) / tf.reduce_mean(1-M)

#%% Solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%%
# Output Initialization
if not os.path.exists('Multiple_Impute_out1/'):
    os.makedirs('Multiple_Impute_out1/')
    
# Iteration Initialization
i = 1

#%% Start Iterations
for it in tqdm(range(10000)):    
    
    #%% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx,:]  
    Z_mb = sample_Z(mb_size, Dim) 
    M_mb = trainM[mb_idx,:]  
    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
    H_mb = M_mb * H_mb1
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                       feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb})
            
    #%% Output figure
    if it % 100 == 0:
      
        mb_idx = sample_idx(Test_No, 5)
        X_mb = testX[mb_idx,:]
        M_mb = testM[mb_idx,:]  
        Z_mb = sample_Z(5, Dim) 
    
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        
        samples1 = X_mb                
        samples5 = M_mb * X_mb + (1-M_mb) * Z_mb
        
        samples2 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
        samples2 = M_mb * X_mb + (1-M_mb) * samples2        
        
        Z_mb = sample_Z(5, Dim) 
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
        samples3 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
        samples3 = M_mb * X_mb + (1-M_mb) * samples3     
        
        Z_mb = sample_Z(5, Dim) 
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
        samples4 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
        samples4 = M_mb * X_mb + (1-M_mb) * samples4     
        
        samples = np.vstack([samples5, samples2, samples3, samples4, samples1])          
        
        fig = plot(samples)
        plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        
    #%% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(MSE_train_loss_curr))
        print('Test_loss: {:.4}'.format(MSE_test_loss_curr))
        print()