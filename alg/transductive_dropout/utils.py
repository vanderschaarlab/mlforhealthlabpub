import autograd.numpy as np
import matplotlib.pyplot as plt

def nice_plot(model,samples_p,y_samples_p,samples_q,y_samples_q,true_func,save = None,params=None):
    '''
    Creates nice plot of predictive distribution given a model.
    '''
    x = np.linspace(0, 18,1000)
    if params:
        model.params = params 
    ns =100
    sample = []
    for _ in range(ns):
        y = model.forward_pass2(model.params,np.reshape(x,(1000,1)))
        sample.append(y)
    sample = np.array(sample)
    sample = np.reshape(sample,(ns,1000))
    s_mean = sample.mean(axis=0)
    s_sd = sample.std(axis=0)

    plt.plot(x, s_mean, linewidth=2, c='royalblue', label='Mean prediction')
    plt.fill_between(np.reshape(x,(1000)), (s_mean - 2 * s_sd), (s_mean + 2 * s_sd),\
        color='cornflowerblue', alpha=.5, label='+/- 2 std')
    plt.scatter(samples_p,y_samples_p, label = 'Training points')
    plt.plot(x, true_func, c = 'r', label = 'True function')
    plt.scatter(samples_q,y_samples_q, label = 'Q points')

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    
    
def nice_plot_conc(model,samples_p,y_samples_p,samples_q,y_samples_q,true_func):
    '''
    Creates nice plot of predictive distribution given a model. Optionally
    pass a new dropout rate to be used.
    '''
    x = np.linspace(0, 18,1000)
    ns =100
    sample = []
    for _ in range(ns):
        y = model.forward_pass(model.params,np.reshape(x,(1000,1)))
        sample.append(y)
    sample = np.array(sample)
    sample = np.reshape(sample,(ns,1000))
    s_mean = sample.mean(axis=0)
    s_sd = sample.std(axis=0)

    plt.plot(x, s_mean, linewidth=2, c='royalblue', label='Mean prediction')
    plt.fill_between(np.reshape(x,(1000)), (s_mean - 2 * s_sd), (s_mean + 2 * s_sd),\
        color='cornflowerblue', alpha=.5, label='+/- 2 std')
    plt.scatter(samples_p,y_samples_p, label = 'Training points')
    plt.plot(x, true_func, c = 'r', label = 'True function')
    plt.scatter(samples_q,y_samples_q, label = 'Q points')

    plt.legend()
    plt.tight_layout()
    plt.show()