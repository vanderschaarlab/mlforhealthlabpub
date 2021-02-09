'''
GANITE: 
Jinsung Yoon 10/11/2017
'''
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
import pandas as pd
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--kk", default=10, type=int)
    parser.add_argument("--it", default=10000, type=int)
    parser.add_argument("-o", default='./result.json')
    parser.add_argument('-ocsv')
    parser.add_argument("--trainx", default="trainx.csv")
    parser.add_argument("--trainy", default="trainy.csv")
    parser.add_argument("--traint")
    parser.add_argument("--testx", default="testx.csv")
    parser.add_argument("--testy", default="testy.csv")
    parser.add_argument("--testt")
    return parser.parse_args()


#%% Performance Metrics
def Perf_RPol_ATT(Test_T, Test_Y, Output_Y):
    # RPol
    # Decision of Output_Y
    hat_t = np.sign(Output_Y[:,1] - Output_Y[:,0])
    hat_t = (0.5*(hat_t + 1))
    new_hat_t = np.abs(1-hat_t)
    
    # Intersection
    idx1 = hat_t * Test_T
    idx0 = new_hat_t  * (1-Test_T)    

    # RPol Computation
    RPol1 = (np.sum(idx1 * Test_Y)/(np.sum(idx1)+1e-8)) * np.mean(hat_t)  
    RPol0 = (np.sum(idx0 * Test_Y)/(np.sum(idx0)+1e-8)) * np.mean(new_hat_t) 
    RPol = 1 - (RPol1 + RPol0)    
    
    # ATT   
    # Original ATT
    ATT_value = np.sum(Test_T * Test_Y) / (np.sum(Test_T) + 1e-8) - np.sum((1-Test_T) * Test_Y) / (np.sum(1-Test_T) + 1e-8)
    # Estimated ATT
    ATT_estimate = np.sum(Test_T * (Output_Y[:,1] - Output_Y[:,0]) ) / (np.sum(Test_T) + 1e-8) 
    # Final ATT    
    ATT = np.abs( ATT_value - ATT_estimate )
    print('pol0:{} pol1:{} pol:{} mean hat:{} mean new hat:{} ATT:{}'.format(RPol0, RPol1, RPol, np.mean(hat_t), np.mean(new_hat_t), ATT))
    return [RPol, ATT]


def PEHE(y, hat_y):
    e_PEHE = tf.reduce_mean( tf.squared_difference( (y[:,1]-y[:,0]), (hat_y[:,1] - hat_y[:,0]) ))
    return e_PEHE


def ATE(y, hat_y):
    e_PEHE = tf.abs( tf.reduce_mean( y[:,1]-y[:,0] ) - tf.reduce_mean( hat_y[:,1]-hat_y[:,0] ) )
    return e_PEHE


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)


# 3.1 Generator
def generator(x,t,y):
    inputs = tf.concat(axis = 1, values = [x,t,y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    
    G_h31 = tf.nn.relu(tf.matmul(G_h2, G_W31) + G_b31)
    G_prob1 = (tf.matmul(G_h31, G_W32) + G_b32)
    
    G_h41 = tf.nn.relu(tf.matmul(G_h2, G_W41) + G_b41)
    G_prob2 = (tf.matmul(G_h41, G_W42) + G_b42)
    
    G_prob = tf.nn.sigmoid(tf.concat(axis = 1, values = [G_prob1, G_prob2]))
    
    return G_prob
    

# 3.2. Discriminator
def discriminator(x,t,y,hat_y):
    # Factual & Counterfactual outcomes concatenate
    inp0 = (1.-t) * y + t * tf.reshape(hat_y[:,0], [-1,1])
    inp1 = t * y + (1.-t) * tf.reshape(hat_y[:,1], [-1,1])
    
    inputs = tf.concat(axis = 1, values = [x,inp0,inp1])
    
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    
    return D_logit


# 3.3. Inference Nets
def inference(x):
    I_h1 = tf.nn.relu(tf.matmul(x, I_W1) + I_b1)
    I_h2 = tf.nn.relu(tf.matmul(I_h1, I_W2) + I_b2)
    
    I_h31 = tf.nn.relu(tf.matmul(I_h2, I_W31) + I_b31)
    I_prob1 = (tf.matmul(I_h31, I_W32) + I_b32)
    
    I_h41 = tf.nn.relu(tf.matmul(I_h2, I_W41) + I_b41)
    I_prob2 = (tf.matmul(I_h41, I_W42) + I_b42)
    
    I_prob = tf.nn.sigmoid(tf.concat(axis = 1, values = [I_prob1, I_prob2]))
    
    return I_prob


# Random sample generator for Z and R
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size = [m, n])        


def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0], size)
    return start_idx


if __name__ == '__main__':

    args = init_arg()

    fn_trainx, fn_trainy, fn_traint = args.trainx, args.trainy, args.traint
    fn_testx, fn_testy, fn_testt = args.testx, args.testy, args.testt

    Train_X = pd.read_csv(fn_trainx).values
    Train_Y = pd.read_csv(fn_trainy).values
    Train_T = pd.read_csv(fn_traint).values if fn_traint is not None else None

    Test_X = pd.read_csv(fn_testx).values
    Test_Y = pd.read_csv(fn_testy).values
    Test_T = pd.read_csv(fn_testt).values if fn_testt is not None else None

    dim_outcome = Test_Y.shape[1]

    fn_json = args.o
    fn_csv = args.ocsv

    num_iterations = args.it

    mb_size = 256
    alpha = args.alpha
    num_kk = args.kk

    Train_No = len(Train_X)
    Test_No = len(Test_X)

    Dim = len(Train_X[0])
    H_Dim1 = int(Dim)
    H_Dim2 = int(Dim)

    tf.reset_default_graph()


    #%% 1. Input
    # 1.1. Feature (X)
    X = tf.placeholder(tf.float32, shape = [None, Dim])
    # 1.2. Treatment (T)
    T = tf.placeholder(tf.float32, shape = [None, 1])
    # 1.3. Outcome (Y)
    Y = tf.placeholder(tf.float32, shape = [None, 1])
    # 1.6. Test Outcome (Y_T) - Potential outcome

    # Y_T = tf.placeholder(tf.float32, shape = [None, 2]) # Twins
    # Y_T = tf.placeholder(tf.float32, shape = [None, 1]) # Jobs
    Y_T = tf.placeholder(tf.float32, shape = [None, dim_outcome])

    #%% 2. layer construction
    # 2.1 Generator Layer
    G_W1 = tf.Variable(xavier_init([(Dim+2), H_Dim1]))   # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector (Z)
    G_b1 = tf.Variable(tf.zeros(shape = [H_Dim1]))

    G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    G_b2 = tf.Variable(tf.zeros(shape = [H_Dim2]))

    G_W31 = tf.Variable(xavier_init([H_Dim2, H_Dim2]))
    G_b31 = tf.Variable(tf.zeros(shape = [H_Dim2]))   # Output: Estimated Potential Outcomes

    G_W32 = tf.Variable(xavier_init([H_Dim2, 1]))
    G_b32 = tf.Variable(tf.zeros(shape = [1]))   # Output: Estimated Potential Outcomes

    G_W41 = tf.Variable(xavier_init([H_Dim2, H_Dim2]))
    G_b41 = tf.Variable(tf.zeros(shape = [H_Dim2]))   # Output: Estimated Potential Outcomes

    G_W42 = tf.Variable(xavier_init([H_Dim2, 1]))
    G_b42 = tf.Variable(tf.zeros(shape = [1]))   # Output: Estimated Potential Outcomes

    theta_G = [G_W1, G_W2, G_W31, G_W32, G_W41, G_W42, G_b1, G_b2, G_b31, G_b32, G_b41, G_b42]

    # 2.2 Discriminator
    D_W1 = tf.Variable(xavier_init([(Dim+2), H_Dim1]))   # Inputs: X + Factual Outcomes + Estimated Counterfactual Outcomes
    D_b1 = tf.Variable(tf.zeros(shape = [H_Dim1]))

    D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    D_b2 = tf.Variable(tf.zeros(shape = [H_Dim2]))

    D_W3 = tf.Variable(xavier_init([H_Dim2, 1]))
    D_b3 = tf.Variable(tf.zeros(shape = [1]))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # 2.3 Inference Layer
    I_W1 = tf.Variable(xavier_init([(Dim), H_Dim1]))
    I_b1 = tf.Variable(tf.zeros(shape = [H_Dim1]))

    I_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    I_b2 = tf.Variable(tf.zeros(shape = [H_Dim2]))

    I_W31 = tf.Variable(xavier_init([H_Dim2, H_Dim2]))
    I_b31 = tf.Variable(tf.zeros(shape = [H_Dim2]))

    I_W32 = tf.Variable(xavier_init([H_Dim2, 1]))
    I_b32 = tf.Variable(tf.zeros(shape = [1]))

    I_W41 = tf.Variable(xavier_init([H_Dim2, H_Dim2]))
    I_b41 = tf.Variable(tf.zeros(shape = [H_Dim2]))

    I_W42 = tf.Variable(xavier_init([H_Dim2, 1]))
    I_b42 = tf.Variable(tf.zeros(shape = [1]))

    theta_I = [I_W1, I_W2, I_W31, I_W32, I_W41, I_W42, I_b1, I_b2, I_b31, I_b32, I_b41, I_b42]


    #%% Structure
    # 1. Generator
    Tilde = generator(X,T,Y)
    # 2. Discriminator
    D_logit = discriminator(X,T,Y,Tilde)
    # 3. Inference function
    Hat = inference(X)

    #%% Loss
    # 1. Discriminator loss
    #D_loss = -tf.reduce_mean(T * tf.log(D_prob + 1e-8) + (1. -T) * tf.log(1. - D_prob + 1e-8)) 
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = T, logits = D_logit )) 

    # 2. Generator loss
    G_loss_GAN = -D_loss 

    G_loss_R = tf.reduce_mean(tf.losses.mean_squared_error(Y, (T * tf.reshape(Tilde[:,1],[-1,1]) + (1. - T) * tf.reshape(Tilde[:,0],[-1,1]) ))) 

    G_loss = G_loss_R + alpha * G_loss_GAN

    # 4. Inference loss

    I_loss1 = tf.reduce_mean(tf.losses.mean_squared_error((T) * Y + (1-T) * tf.reshape(Tilde[:,1],[-1,1]), tf.reshape(Hat[:,1],[-1,1]) )) 
    I_loss2 = tf.reduce_mean(tf.losses.mean_squared_error((1-T) * Y +  (T) * tf.reshape(Tilde[:,0],[-1,1]), tf.reshape(Hat[:,0],[-1,1]) )) 

    I_loss = I_loss1 + I_loss2

    # Loss Followup    
    if Test_T is None:
        Hat_Y = Hat
        Loss1 = PEHE(Y_T, Hat_Y)
        Loss2 = ATE(Y_T, Hat_Y)

    #%% Solver
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    I_solver = tf.train.AdamOptimizer().minimize(I_loss, var_list=theta_I)

    #%% Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Iterations
    # Train G and D first
    for it in tqdm(range(num_iterations)):

        for kk in range(num_kk):
            idx_mb = sample_X(Train_X, mb_size) 
            X_mb = Train_X[idx_mb,:]
            T_mb = np.reshape(Train_T[idx_mb], [mb_size,1]) 
            Y_mb = np.reshape(Train_Y[idx_mb], [mb_size,1])   

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb})

        idx_mb = sample_X(Train_X, mb_size) 
        X_mb = Train_X[idx_mb,:]
        T_mb = np.reshape(Train_T[idx_mb], [mb_size,1]) 
        Y_mb = np.reshape(Train_Y[idx_mb], [mb_size,1])   

        _, G_loss_curr, Tilde_curr = sess.run([G_solver, G_loss, Tilde], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb})

        #%% Testing
        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D_loss: {:.4}'.format((D_loss_curr)))        
            print('G_loss: {:.4}'.format((G_loss_curr)))    
            print()

    # Train I and ID
    result = {}
    for it in tqdm(range(num_iterations)):

        idx_mb = sample_X(Train_X, mb_size)
        X_mb = Train_X[idx_mb,:]
        T_mb = np.reshape(Train_T[idx_mb], [mb_size,1]) 
        Y_mb = np.reshape(Train_Y[idx_mb], [mb_size,1])   

        _, I_loss_curr = sess.run([I_solver, I_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb})    

        #%% Testing
        if it % 100 == 0:
            result = {
                'alpha': alpha,
                'kk': num_kk
            }
            if Test_T is not None:
                Hat_curr = sess.run([Hat], feed_dict = {X: Test_X})[0]      
                [R_Pol_Out, B] = Perf_RPol_ATT(Test_T, Test_Y, Hat_curr)

                print('Iter: {}'.format(it)) 
                print('I_loss: {:.4}'.format((I_loss_curr)))      
                print('R_Pol_Out: {:.4}'.format(R_Pol_Out))    
                print('')
                result['R_Pol_Out'] = float(R_Pol_Out)
            else:
                New_X_mb = Test_X 
                Y_T_mb = Test_Y           

                Loss1_curr, Loss2_curr, Hat_curr = sess.run([Loss1, Loss2, Hat], feed_dict = {X: New_X_mb, Y_T: Y_T_mb})        

                print('Iter: {}'.format(it)) 
                print('I_loss: {:.4}'.format((I_loss_curr)))      
                print('Loss_PEHE_Out: {:.4}'.format(np.sqrt(Loss1_curr)))   
                print('Loss_ATE_Out: {:.4}'.format(Loss2_curr))   
                print('')
                result['Loss_PEHE_Out'] = float(np.sqrt(Loss1_curr))
                result['Loss_ATE_Out'] = float(Loss2_curr)

    with open(fn_json, "w") as fp:
        json.dump(result, fp)

    if fn_csv is not None:
        Hat_curr = sess.run([Hat], feed_dict = {X: Test_X})[0]
        if Test_T is not None:
            [R_Pol_Out, B] = Perf_RPol_ATT(Test_T, Test_Y, Hat_curr)
        df = pd.DataFrame(Hat_curr, columns=['A', 'B'])
        df.to_csv(fn_csv, index=False)

        odir = os.path.dirname(fn_csv)

        df_test_X = pd.DataFrame(Test_X)
        df_test_X.to_csv('{}/testx.csv'.format(odir), index=False)

        df_test_Y = pd.DataFrame(Test_Y)
        df_test_Y.to_csv('{}/testy.csv'.format(odir), index=False)

        if Test_T is not None:
            df_test_T = pd.DataFrame(Test_T)
            fn_test1 = '{}/testt.csv'.format(odir)
            df_test_T.to_csv(fn_test1, index=False)
