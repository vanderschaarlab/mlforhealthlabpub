import numpy as np
np.set_printoptions(suppress=True)
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from sklearn.metrics import mean_squared_error

# this allows wider numpy viewing for matrices
np.set_printoptions(linewidth=np.inf)

class CASTLE(object):
    def __init__(self, num_train, lr  = None, batch_size = 32, num_inputs = 1, num_outputs = 1,
                 w_threshold = 0.3, n_hidden = 32, hidden_layers = 2, ckpt_file = 'tmp.ckpt',
                 standardize = True,  reg_lambda=None, reg_beta=None, DAG_min = 0.5):
        
        self.w_threshold = w_threshold
        self.DAG_min = DAG_min
        if lr is None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = lr

        if reg_lambda is None:
            self.reg_lambda = 1.
        else:
            self.reg_lambda = reg_lambda

        if reg_beta is None:
            self.reg_beta = 1
        else:
            self.reg_beta = reg_beta

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.X = tf.placeholder("float", [None, self.num_inputs])
        self.y = tf.placeholder("float", [None, 1])
        self.rho =  tf.placeholder("float",[1,1])
        self.alpha =  tf.placeholder("float",[1,1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.count = 0
        self.max_steps = 200
        self.saves = 50 
        self.patience = 30
        self.metric = mean_squared_error

        
        # One-hot vector indicating which nodes are trained
        self.sample =tf.placeholder(tf.int32, [self.num_inputs])
        
        # Store layers weight & bias
        seed = 1
        self.weights = {}
        self.biases = {}
        
        # Create the input and output weight matrix for each feature
        for i in range(self.num_inputs):
            self.weights['w_h0_'+str(i)] = tf.Variable(tf.random_normal([self.num_inputs, self.n_hidden], seed = seed)*0.01)
            self.weights['out_'+str(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed = seed))
            
        for i in range(self.num_inputs):
            self.biases['b_h0_'+str(i)] = tf.Variable(tf.random_normal([self.n_hidden], seed = seed)*0.01)
            self.biases['out_'+str(i)] = tf.Variable(tf.random_normal([self.num_outputs], seed = seed))
        
        
        # The first and second layers are shared
        self.weights.update({
            'w_h1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        })
        
        
        self.biases.update({
            'b_h1': tf.Variable(tf.random_normal([self.n_hidden]))
        })
        
            
        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.layer_1 = {}
        self.layer_1_dropout = {}
        self.out_layer = {}
       
        self.Out_0 = []
        
        # Mask removes the feature i from the network that is tasked to construct feature i
        self.mask = {}
        self.activation = tf.nn.relu
            
        for i in range(self.num_inputs):
            indices = [i]*self.n_hidden
            self.mask[str(i)] = tf.transpose(tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1))
            
            self.weights['w_h0_'+str(i)] = self.weights['w_h0_'+str(i)]*self.mask[str(i)] 
            self.hidden_h0['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.X, self.weights['w_h0_'+str(i)]), self.biases['b_h0_'+str(i)]))
            self.hidden_h1['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.hidden_h0['nn_'+str(i)], self.weights['w_h1']), self.biases['b_h1']))
            self.out_layer['nn_'+str(i)] = tf.matmul(self.hidden_h1['nn_'+str(i)], self.weights['out_'+str(i)]) + self.biases['out_'+str(i)]
            self.Out_0.append(self.out_layer['nn_'+str(i)])
        
        # Concatenate all the constructed features
        self.Out = tf.concat(self.Out_0,axis=1)
        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
           
        self.supervised_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out_layer['nn_0'] - self.y),axis=1),axis=0)
        self.regularization_loss = 0

        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(tf.math.sqrt(tf.reduce_sum(tf.square(self.weights['w_h0_'+str(i)]),axis=1,keepdims=True)))
        
        self.W = tf.concat(self.W_0,axis=1)
               
        #truncated power series
        d = tf.cast(self.X.shape[1], tf.float32)
        coff = 1.0 
        Z = tf.multiply(self.W,self.W)
       
        dag_l = tf.cast(d, tf.float32) 
       
        Z_in = tf.eye(d)
        for i in range(1,10):
           
            Z_in = tf.matmul(Z_in, Z)
           
            dag_l += 1./coff * tf.linalg.trace(Z_in)
            coff = coff * (i+1)
        
        self.h = dag_l - tf.cast(d, tf.float32)

        # Residuals
        self.R = self.X - self.Out 
        # Average reconstruction loss
        self.average_loss = 0.5 / num_train * tf.reduce_sum(tf.square(self.R))


        #group lasso
        L1_loss = 0.0
        for i in range(self.num_inputs):
            w_1 = tf.slice(self.weights['w_h0_'+str(i)],[0,0],[i,-1])
            w_2 = tf.slice(self.weights['w_h0_'+str(i)],[i+1,0],[-1,-1])
            L1_loss += tf.reduce_sum(tf.norm(w_1,axis=1))+tf.reduce_sum(tf.norm(w_2,axis=1))
        
        # Divide the residual into untrain and train subset
        _, subset_R = tf.dynamic_partition(tf.transpose(self.R), partitions=self.sample, num_partitions=2)
        subset_R = tf.transpose(subset_R)

        #Combine all the loss
        self.mse_loss_subset = tf.cast(self.num_inputs, tf.float32)/ tf.cast(tf.reduce_sum(self.sample), tf.float32)* tf.reduce_sum(tf.square(subset_R))
        self.regularization_loss_subset =  self.mse_loss_subset +  self.reg_beta * L1_loss +  0.5 * self.rho * self.h * self.h + self.alpha * self.h
            
        #Add in supervised loss
        self.regularization_loss_subset +=  self.Lambda *self.rho* self.supervised_loss
        self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)


        self.loss_op_supervised = self.optimizer_subset.minimize(self.supervised_loss + self.regularization_loss)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())     
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file
        
    def __del__(self):
        tf.reset_default_graph()
        print("Destructor Called... Cleaning up")
        self.sess.close()
        del self.sess
        
    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
        return input_layer + noise
    
    
    def fit(self, X, y,num_nodes, X_val, y_val, X_test, y_test):         
        
        from random import sample 
        rho_i = np.array([[1.0]])
        alpha_i = np.array([[1.0]])
        
        best = 1e9
        best_value = 1e9
        for step in range(1, self.max_steps):
            h_value, loss = self.sess.run([self.h, self.supervised_loss], feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
            print("Step " + str(step) + ", Loss= " + "{:.4f}".format(loss)," h_value:", h_value) 

                
            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):

               
                idxs = random.sample(range(X.shape[0]), self.batch_size)
                batch_x = X[idxs]
                batch_y = np.expand_dims(batch_x[:,0], -1)
                one_hot_sample = [0]*self.num_inputs
                subset_ = sample(range(self.num_inputs),num_nodes) 
                for j in subset_:
                    one_hot_sample[j] = 1
                self.sess.run(self.loss_op_dag, feed_dict={self.X: batch_x, self.y: batch_y, self.sample:one_hot_sample,
                                                              self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.Lambda : self.reg_lambda, self.is_train : True, self.noise : 0})

            val_loss = self.val_loss(X_val, y_val)
            if val_loss < best_value:
                best_value = val_loss
            h_value, loss = self.sess.run([self.h, self.supervised_loss], feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
            if step >= self.saves:
                try:
                    if val_loss < best:
                        best = val_loss 
                        self.saver.save(self.sess, self.tmp)
                        print("Saving model")
                        self.count = 0
                    else:
                        self.count += 1
                except:
                    print("Error caught in calculation")
                    
            if self.count > self.patience:
                print("Early stopping")
                break

        self.saver.restore(self.sess, self.tmp)
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:rho_i, self.alpha:alpha_i, self.is_train : True, self.noise:0})
        W_est[np.abs(W_est) < self.w_threshold] = 0

   
    def val_loss(self, X, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, -1)
        from random import sample 
        one_hot_sample = [0]*self.num_inputs
        
        # use all values for validation
        subset_ = sample(range(self.num_inputs),self.num_inputs) 
        for j in subset_:
            one_hot_sample[j] = 1
        
        return self.sess.run(self.supervised_loss, feed_dict={self.X: X, self.y: y, self.sample:one_hot_sample, self.keep_prob : 1, self.rho:np.array([[1.0]]), 
                                                              self.alpha:np.array([[0.0]]), self.Lambda : self.reg_lambda, self.is_train : False, self.noise:0})
        
        
    def pred(self, X):
        return self.sess.run(self.out_layer['nn_0'], feed_dict={self.X: X, self.keep_prob:1, self.is_train : False, self.noise:0})
        
    def get_weights(self, X, y):
        return self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
    
    def pred_W(self, X, y):
        W_est = self.sess.run(self.W, feed_dict={self.X: X, self.y: y, self.keep_prob : 1, self.rho:np.array([[1.0]]), self.alpha:np.array([[0.0]]), self.is_train : False, self.noise:0})
        return np.round_(W_est,decimals=3)
