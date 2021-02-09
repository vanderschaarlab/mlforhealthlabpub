import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

def bias_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

# Get networks
def get_network(network_name, Learning_rate, dense1, dense2, dense_mu, dense_sig, input_dim):
    input_x = tf.placeholder(tf.float64, shape = [None, input_dim])
    
    with tf.variable_scope(network_name):
        # Densely connect layer variables
        w_fc1 = weight_variable(network_name + '_w_fc1', dense1)
        b_fc1 = bias_variable(network_name + '_b_fc1', [dense1[1]])
        
        w_fc2 = weight_variable(network_name + '_w_fc2', dense2)
        b_fc2 = bias_variable(network_name + '_b_fc2', [dense2[1]])
        
        w_fc_mu = weight_variable(network_name + '_w_fc_mu', dense_mu)
        b_fc_mu = bias_variable(network_name + '_b_fc_mu', [dense_mu[1]])

        w_fc_sig = weight_variable(network_name + '_w_fc_sig', dense_sig)
        b_fc_sig = bias_variable(network_name + '_b_fc_sig', [dense_sig[1]])

    # Network
    fc1 = tf.nn.relu(tf.matmul(input_x, w_fc1) + b_fc1)
    fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc2)
    output_mu  = tf.matmul(fc2, w_fc_mu) + b_fc_mu
    output_sig = tf.matmul(fc2, w_fc_sig) + b_fc_sig
    output_sig_pos = tf.log(1 + tf.exp(output_sig)) + 1e-06
    
    y = tf.placeholder(tf.float64, shape = [None, 1])
    
    # Negative Log Likelihood(NLL) 
    loss = tf.reduce_mean(0.5*tf.log(output_sig_pos) + 0.5*tf.div(tf.square(y - output_mu),output_sig_pos)) + 10
  
    # Get trainable variables
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name) 
  
    # Gradient clipping for preventing nan
    optimizer = tf.train.AdamOptimizer(learning_rate = Learning_rate)
    gvs = optimizer.compute_gradients(loss, var_list = train_vars)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_opt = optimizer.apply_gradients(capped_gvs)

    return input_x, y, output_mu, output_sig_pos, loss, train_opt, train_vars


# Make batch data 
def making_batch(data_size, sample_size, data_x, data_y):
    
    # Making batches(testing)
    batch_idx = np.random.choice(data_size, sample_size)
    
    batch_x = np.zeros([sample_size, data_x.shape[1]])
    batch_y = np.zeros([sample_size, 1])
        
    for i in range(batch_idx.shape[0]):
        batch_x[i,:] = data_x[batch_idx[i], :]
        batch_y[i,:] = data_y[batch_idx[i], :] 
        
    return batch_x, batch_y


# In[19]:

def deep_ensemble(X, Y, X1, input_dim=1, Learning_rate=0.01, epsilon=1e-8, 
                  num_iter=1000, batch_size=50, n_hidden=100, verbose=False):
    
    # Ensemble networks (5 networks)
    networks      = ['network1', 'network2', 'network3', 'network4', 'network5']
    
    # Dense [input size, output size]
    dense1        = [input_dim, n_hidden]
    dense2        = [n_hidden, n_hidden]
    dense_mu      = [n_hidden, 1]
    dense_sig     = [n_hidden, 1]
    
    #######
    
    tf.reset_default_graph()
    
    x_list           = []
    y_list           = []
    output_mu_list   = []
    output_sig_list  = []
    loss_list        = []
    train_list       = []
    train_var_list   = []
    output_test_list = []

    # Train each ensemble network
    for i in range(len(networks)):
        x_input, y, output_mu, output_sig, loss, train_opt, train_vars = get_network(networks[i], Learning_rate, dense1, 
                                                                                     dense2, dense_mu, dense_sig, input_dim)

        x_list.append(x_input)
        y_list.append(y)
        output_mu_list.append(output_mu)
        output_sig_list.append(output_sig)
        loss_list.append(loss)
        train_list.append(train_opt)
        train_var_list.append(train_vars)
    
    # Create Session
    config = tf.ConfigProto()

    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    
    
    train_data_num = X.shape[0]

    loss_train     = np.zeros([len(networks)])
    
    for iter in range(num_iter):
        
        for i in range(len(networks)):
            # Making batches(training)
            batch_x, batch_y = making_batch(train_data_num, batch_size, X, Y)
       
            # Training
            _, loss, mu, sig = sess.run([train_list[i], loss_list[i], output_mu_list[i], output_sig_list[i]], 
                                     feed_dict = {x_list[i]: batch_x, y_list[i]: batch_y})
  

            #if np.any(np.isnan(loss)):
            #    raise ValueError('There is Nan in loss')
        
            loss_train[i] += loss
    
    ## Testing
                
    # output for ensemble network
    out_mu_sample  = np.zeros([X1.shape[0], len(networks)])
    out_sig_sample = np.zeros([X1.shape[0], len(networks)])


    for i in range(len(networks)):
        mu_sample, sig_sample = sess.run([output_mu_list[i], output_sig_list[i]], 
                                      feed_dict = {x_list[i]: X1})

        out_mu_sample[:,i]  = np.reshape(mu_sample, (X1.shape[0]))
        out_sig_sample[:,i] = np.reshape(sig_sample, (X1.shape[0]))

    out_mu_sample_final  = np.mean(out_mu_sample, axis = 1)
    out_sig_sample_final = np.sqrt(np.mean(out_sig_sample + np.square(out_mu_sample), axis = 1) - np.square(out_mu_sample_final))


    return out_mu_sample_final, out_sig_sample_final 

