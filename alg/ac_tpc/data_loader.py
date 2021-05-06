import numpy as np


def import_data(data_name = 'sample'):
    '''
        Output: 
            - data_x: [N, max_length, 1+x_dim] tensor (where N: number of samples, max_length: max sequence length, x_dim: feature dimension)
                      the first feature is the time difference.
            - data_y: [N, max_length, y_dim] tensor (where N: number of samples, max_length: max sequence length, y_dim: output dimension)
    '''
    if data_name == 'sample':
        npz    = np.load('./data/sample/data.npz')
        data_x = npz['data_x']
        data_y = npz['data_y']
        y_type = npz['y_type'] #{'binary', 'categorical', 'continuous'}
    else:
        raise ValueError('error: data_name not defined')

    return data_x, data_y, y_type