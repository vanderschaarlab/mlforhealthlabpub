import pickle


def write_results_to_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

def append_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)



