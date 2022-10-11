import matplotlib.pyplot as plt
import pickle


'''
--------------------------------------------------------------------------------------------
Generate two saples of random functions observed at irregularly sampled times
--------------------------------------------------------------------------------------------
'''

def generate_random_functions(size=1000, num_obs = 5, function= 'sine',variance = 0.1, error='gaussian'):
    '''Generate samples from a random functions

    Arguments:
        size : number of samples
        nun_obs: number of observations in each trajectory
        function: specify function for the mean trend

    Output:
        array of times and array of observations at those times
    '''

    np.random.seed()

    Y = np.zeros(num_obs) # placeholder for observation values
    T = np.zeros(num_obs) # placeholder for observation times

    for i in range(size):
        t = np.random.uniform(size=num_obs,low=0,high=1)
        t = np.sort(t)

        if function == 'sine':
            mu = 0.1*np.sin(2*pi*t)
        if function == 'zero':
            mu = np.zeros(len(t))
        if function == 'spike':
            mu = np.exp(-(t-0.5)**2/0.005)

        if error == 'gaussian':
            cov = np.diag(np.ones(len(t))*variance)
            y = np.random.multivariate_normal(mu, cov)

        if error == 'laplace':
            y = mu + np.random.laplace(loc=0,scale = variance, size=len(mu))

        Y = np.vstack((Y, y))
        T = np.vstack((T, t))

    return np.array(T[1:]), np.array(Y[1:])

'''
--------------------------------------------------------------------------------------------
Import Cystic Fibrosis Data for analysis
--------------------------------------------------------------------------------------------
'''

def generate_CF_samples(outcome = 'FEV', feature = 'Gender'):
    '''
    :param outcome: choose outcome, either FEV1 or FEV1PREDICTED
    :param feature: feature to split the data into two samples
    :return: sets of times and observed values
    '''
    if outcome == 'FEV1':
        with open ('/home/alexis/Documents/Datasets/Cystic Fibrosis/FEV1', 'rb') as fp:
            outcome = pickle.load(fp)
    else:
        with open ('/home/alexis/Documents/Datasets/Cystic Fibrosis/FEV1_PREDICTED', 'rb') as fp:
            outcome = pickle.load(fp)

    with open ('/home/alexis/Documents/Datasets/Cystic Fibrosis/TIMES', 'rb') as fp:
        TIMES = pickle.load(fp)
    with open('/home/alexis/Documents/Datasets/Cystic Fibrosis/BASELINE', 'rb') as fp:
        BASELINE = pickle.load(fp)



    idx_0 = [i for i, e in enumerate(BASELINE[feature]) if e == 0]
    idx_1 = [i for i, e in enumerate(BASELINE[feature]) if e == 1]

    t0, t1 = [TIMES[i] for i in idx_0], [TIMES[i]  for i in idx_1]
    y0, y1 = [outcome[i]  for i in idx_0], [outcome[i]  for i in idx_1]

    return t0, y0, t1, y1


'''
--------------------------------------------------------------------------------------------
Import Climate Change Data for analysis
--------------------------------------------------------------------------------------------
'''

def generate_climate_samples(country='US'):
    '''
    :param country: choies are: 'UK', 'US', 'VT', 'TZ', 'PL', 'CA', 'NG', 'PR'
    :return: reference temperatures from the 1850s and list of arrays with
    subsequent periods of 20 year data
    '''
    DataPath = '/home/alexis/Documents/Datasets/Climate/temperatures_' + country + '.csv'
    temperatures = pd.read_csv(DataPath,index_col=0).values
    temperatures_ref = temperatures[:20]

    updated_temp = temperatures[(len(temperatures)-160):]
    temperatures_test = [updated_temp[20:40], updated_temp[40:60], updated_temp[60:80],
                         updated_temp[80:100], updated_temp[100:120], updated_temp[120:140],
                         updated_temp[140:160]]

    return temperatures_ref, temperatures_test

def generate_peru_samples(country='PR',n_samples=20):
    '''
    :param country: choies are: 'UK', 'US', 'VT', 'TZ', 'PL', 'CA', 'NG', 'PR'
    :return: reference temperatures from the 1850s and list of arrays with
    subsequent periods of 20 year data
    '''
    DataPath = '/home/alexis/Documents/Datasets/Climate/temperatures_' + country + '.csv'
    temperatures = pd.read_csv(DataPath,index_col=0).values
    temperatures_ref = temperatures[:20]

    updated_temp = temperatures[(len(temperatures)-100):]
    temperatures_test = updated_temp[random.sample(range(20, 100), n_samples)]

    return temperatures_ref, temperatures_test