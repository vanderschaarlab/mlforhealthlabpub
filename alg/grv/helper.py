import numpy as np
import tensorflow as tf






def syn_data_4(n_samples):

    x_1 = np.random.normal(loc=0.0, scale=1.0, size=[n_samples, 50])
    x_2 = np.zeros([n_samples, 50])

    b_1 = np.random.binomial(n=1,p=0.5,size=[n_samples,1])
    a_1 = np.concatenate([1-b_1, b_1],axis=1)
    b_2 = np.random.binomial(n=1,p=0.5,size=[n_samples,1])
    a_2 = np.concatenate([1-b_2, b_2],axis=1)
    b_1[b_1<0.5] = -1
    b_2[b_2<0.5] = -1


    y_1 = b_1[:,0]* (1+ 1.5*x_1[:,2])  + np.random.normal(loc=0.0, scale=1.0, size=[n_samples,])
    x_2[:,0] = 1.25*x_1[:,0]*b_1[:,0] +  np.random.normal(loc=0.0, scale=1.0, size=[n_samples,])
    x_2[:,0] = (x_2[:,0]>0)*1.0
    x_2[:,1] = -1.75*x_1[:,1]*b_1[:,0] +  np.random.normal(loc=0.0, scale=1.0, size=[n_samples,])
    x_2[:,1] = (x_2[:,1]>0)*1.0
    x_2[:,2]  = y_1
    y_2 = (0.5 + y_1 + 0.5*b_1[:,0] + 0.5*x_2[:,0] - 0.5*x_2[:,1]) * b_2[:,0] + np.random.normal(loc=0.0, scale=1.0, size=[n_samples,])

    x_list = np.concatenate([x_1[:,None],x_2[:,None]],axis=1)
    y_list = np.concatenate([y_1[:,None],y_2[:,None]],axis=1)
    a_list = np.concatenate([a_1[:,None],a_2[:,None]],axis=1)

    return x_list, a_list, y_list


def t_function_4(x_list, a_list,y_list, step):

    n_samples = np.shape(x_list)[0]
    x_1 = x_list[:, 0, :]
    x_2 = x_list[:, 1, :]
    y_1 = y_list[:,0]
    y_2 = y_list[:,1]

    b_1 = np.argmax(a_list[:,0,:],axis=1)
    b_2 = np.argmax(a_list[:,1,:],axis=1)
    b_1[b_1 < 0.5] = -1
    b_2[b_2 < 0.5] = -1

    if step<1:
        y_1 = b_1 * (1 + 1.5 * x_1[:, 2]) + np.random.normal(loc=0.0, scale=1.0, size=[n_samples,])
        x_2[:, 0] = 1.25 * x_1[:, 0] * b_1 + np.random.normal(loc=0.0, scale=1.0, size=[n_samples, ])
        x_2[:, 0] = (x_2[:, 0] > 0) * 1.0
        x_2[:, 1] = -1.75 * x_1[:, 1] * b_1 + np.random.normal(loc=0.0, scale=1.0, size=[n_samples, ])
        x_2[:, 1] = (x_2[:, 1] > 0) * 1.0
        x_2[:, 2] = y_1

    else:
        y_2 = (0.5 + y_1 + 0.5*b_1 + 0.5*x_2[:,0] - 0.5*x_2[:,1]) * b_2 + np.random.normal(loc=0.0, scale=1.0, size=[n_samples,])

    x_list = np.concatenate([x_1[:, None], x_2[:, None]], axis=1)
    y_list = np.concatenate([y_1[:,None],y_2[:,None]],axis=1)

    return x_list, y_list


def final_test(data_list, data_name, exp_index, print_=True):
    x_list, a_list, y_list = data_list
    n_steps = np.shape(a_list)[1]

    from GRVB import GRV_B
    for time in range(1, n_steps + 1):


        model = GRV_B(x_list, a_list, time=time)
        model.load_network(data_name, exp_index, time)

        prob_t, d_matrix_t = model.predict(rnn_X=x_list, treatments=a_list)
        a_list[:, time - 1, :] = d_matrix_t

        x_list, y_list = t_function(x_list, a_list,y_list,time-1)

        model.destroy_graph()
    if print_:
        print("############### Final results ###############")
        print("Rewards:", np.mean(y_list, axis=0), "Cumulative reward:", np.mean(np.sum(y_list, axis=1), axis=0))
    return np.mean(np.sum(y_list, axis=1), axis=0)


def GRVS_final(data_list,data_name, model, print_=True):

    x_list, a_list, y_list = data_list
    n_steps = np.shape(a_list)[1]

    for current_step in range(n_steps):
        prob, d_matrix = model.predict(rnn_X=x_list, treatments=a_list)

        a_list = d_matrix

        x_list, y_list = t_function(x_list, a_list, y_list, current_step)

    if print_:
        print("Rewards:", np.mean(y_list, axis=0),"Cumulative reward:", np.mean(np.sum(y_list, axis=1), axis=0))

    return y_list


def step_test(data_test,model,time, print_=True):

    x_test, a_test, y_test = data_test
    _, d_matrix_t = model.predict(rnn_X=x_test, treatments=a_test)
    a_test[:, time - 1, :] = d_matrix_t
    x_test, y_test = t_function(x_test, a_test, y_test, time - 1)
    if print_:
        print("Rewards:", np.mean(y_test, axis=0),"Cumulative reward:", np.mean(np.sum(y_test, axis=1), axis=0))


    data_test = x_test, a_test, y_test

    return data_test, np.mean(np.sum(y_test, axis=1), axis=0)


def t_function(x_list, a_list,y_list,time):

    x_list, y_list = t_function_4(x_list, a_list, y_list, time)

    return x_list, y_list


def data_load(n_samples):

    x_list, a_list, y_list = syn_data_4(n_samples)

    return x_list, a_list, y_list


def d_concat(d_t, d_t_T,time, n_steps):

    if time == n_steps:
        d_t_T = np.swapaxes(np.expand_dims(d_t, axis=2), 1, 2)
    else:
        d_t = np.swapaxes(np.expand_dims(d_t, axis=2), 1, 2)
        d_t_T = np.concatenate([d_t, d_t_T], axis=1)

    return d_t_T


def tilde(v,std,mean=0.0):
    return v + np.random.normal(loc=mean, scale=std, size=np.shape(v)[0])


def hist_embedding(output, step):
    last_step = max(step - 1, 0)

    embed_t = tf.squeeze(tf.slice(output, [0, last_step, 0], [-1, 1, -1]), axis=1)

    if step - 1 < 0:
        embed_t = tf.zeros_like(embed_t)

    return embed_t





