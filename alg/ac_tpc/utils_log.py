import tensorflow as tf

### PARAMETER LOGGING
def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            if 'activate_fn' in key:
                value = str(value).split(' ')[1]
                
            f.write('%s:%s\n' % (key, value))


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                
                if 'activate_fn' in key:
                    if value == 'relu':
                        value = tf.nn.relu
                    elif value == 'elu':
                        value = tf.nn.elu
                    elif value == 'tanh':
                        value = tf.nn.tanh
                    else:
                        raise ValueError('ERROR: wrong choice of activation function!')
                    data[key] = value
                else:
                    if value.isdigit():
                        data[key] = int(value)
                    elif is_float(value):
                        data[key] = float(value)
                    elif value == 'None':
                        data[key] = None
                    else:
                        data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data