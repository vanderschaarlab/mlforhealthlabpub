"""
The seeker module containing the `seeker(...)` function.
"""
# pylint: disable=fixme
from typing import Dict
import numpy as np
from utils.data_preprocess import preprocess_data


def seeker(input_dict: Dict) -> np.ndarray:
    """Solution seeker function.

    Args:
        input_dict (Dict): Dictionary that contains the seeker function inputs, as below:
            * "seed" (int): Random seed provided by the competition, use for reproducibility.
            * "generated_data" (np.ndarray of float): Generated dataset from hider data, 
                shape [num_examples, max_seq_len, num_features].
            * "enlarged_data" (np.ndarray of float): Enlarged original dataset, 
                shape [num_examples_enlarge, max_seq_len, num_features].
            * "generated_data_padding_mask" (np.ndarray of bool): Padding mask of bools, generated dataset, 
                same shape as "generated_data".
            * "enlarged_data_padding_mask" (np.ndarray of bool): Padding mask of bools, enlarged dataset, 
                same shape as "enlarged_data".

    Returns:
        np.ndarray: The reidentification labels produced by the seeker, expected shape [num_examples_enlarge].
    """

    # Get the inputs.
    seed = input_dict["seed"]
    generated_data = input_dict["generated_data"]
    enlarged_data = input_dict["enlarged_data"]
    generated_data_padding_mask = input_dict["generated_data_padding_mask"]
    enlarged_data_padding_mask = input_dict["enlarged_data_padding_mask"]

    # Get processed and imputed data, if desired:
    generated_data_preproc, generated_data_imputed = preprocess_data(generated_data, generated_data_padding_mask)
    enlarged_data_preproc, enlarged_data_imputed = preprocess_data(enlarged_data, enlarged_data_padding_mask)

    # TODO: Put your seeker code to replace Example 1 below.
    # Feel free play around with Examples 1 (knn) and 2 (binary_predictor) below.

    # --- Example 1: knn ---
    from examples.seeker.knn import knn_seeker

    reidentified_data = knn_seeker.knn_seeker(generated_data_imputed, enlarged_data_imputed)
    return reidentified_data

    # --- Example 2: binary_predictor ---
    # from utils.misc import tf115_found
    # assert tf115_found is True, "TensorFlow 1.15 not found, which is required to run binary_predictor."
    # from examples.seeker.binary_predictor import binary_predictor
    # reidentified_data = binary_predictor.binary_predictor(generated_data_imputed, enlarged_data_imputed, verbose=True)
    # return generated_data
