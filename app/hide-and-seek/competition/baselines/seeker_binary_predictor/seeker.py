from utils.data_preprocess import preprocess_data
from binary_predictor.binary_predictor import binary_predictor


def seeker(input_dict):

    seed = input_dict["seed"]
    generated_data = input_dict["generated_data"]
    enlarged_data = input_dict["enlarged_data"]
    generated_data_padding_mask = input_dict["generated_data_padding_mask"]
    enlarged_data_padding_mask = input_dict["enlarged_data_padding_mask"]

    print("Competition-provided random seed:", seed)

    generated_data_preproc, generated_data_imputed = preprocess_data(generated_data, generated_data_padding_mask)
    enlarged_data_preproc, enlarged_data_imputed = preprocess_data(enlarged_data, enlarged_data_padding_mask)

    return binary_predictor(generated_data_imputed, enlarged_data_imputed)
