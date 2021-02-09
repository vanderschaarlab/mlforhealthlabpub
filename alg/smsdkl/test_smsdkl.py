from keras import backend as K

from model import SMSDKL

K.tensorflow_backend._get_available_gpus()
import numpy as np
from data_load import load_house
from Evaluate import get_opt_domain, init_random_uniform, evaluate,error_function
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#Set the maximum number of BO iterations
additional_query_size = 51

#Set the hyperparameter space
lim_domain, dim, bounds = get_opt_domain()

#loading the housing dataset
Dataset, dataX, data_label, trainX, testX, label_train, label_test = load_house()


#Uniformly sample one point to start optimization
list_domain = init_random_uniform(lim_domain, n_points=1,initial=True)
obs = evaluate(trainX, testX, label_train, label_test, list_domain)
BO_data = np.array(list_domain), obs



#Set up the BO model
BO_model = SMSDKL(data_X=Dataset, bounds=bounds, BO_data=BO_data,lim_domain=lim_domain)


rmse_global = []
rmse_wise = []

rmse, _ = error_function(obs)
rmse_global.append(rmse)
rmse_wise.append(rmse)


for i in range(additional_query_size):


    #Optimize the BO network
    if i %5==0:
        try:

            BO_model.batch_optimization()

        except Exception:

            print("Covariance matrix not invertible.")

    #Make a acquistion
    query = BO_model.acquisition()

    #Evaluate the acquistion
    obs = evaluate(trainX, testX, label_train, label_test, [query])


    #Updat the acquisition set
    BO_input, BO_output = BO_model.update_data(query, obs)


    print('Number of Function Evaluations:', i)

    print('query point:', query)


    #Compute the minimum rmse obtained by global and step-wise model selection
    rmse1,rmse2 = error_function(BO_output)



    #Compare the convergence plot of global and step-wise model selection
    print('minimum rmse of global model selection:', rmse1)
    print('minimum rmse of step-wise model selection:', rmse2)

    rmse_global.append(rmse1)
    rmse_wise.append(rmse2)

    fig = plt.figure()
    plt.plot(rmse_global, 'darkgreen')
    plt.plot(rmse_wise, 'crimson')
    red_patch = mpatches.Patch(color='crimson', label='Step-wise')
    green_patch = mpatches.Patch(color='darkgreen', label='Global')
    plt.legend(handles=[green_patch,red_patch])
    plt.title('Dataset: Housing')
    plt.xlabel('BO Iterations')
    plt.ylabel('rmse')
    fig.savefig('plot.png', dpi=300)
    plt.close()
