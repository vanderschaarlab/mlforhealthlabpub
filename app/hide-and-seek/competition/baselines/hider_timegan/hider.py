from utils.data_preprocess import preprocess_data
from timegan.timegan import timegan


def hider(input_dict):

    seed = input_dict["seed"]
    data = input_dict["data"]
    padding_mask = input_dict["padding_mask"]

    print("Competition-provided random seed:", seed)

    data_preproc, data_imputed = preprocess_data(data, padding_mask)

    return timegan(data_imputed)
