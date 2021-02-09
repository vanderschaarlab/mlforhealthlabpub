"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com
"""
from typing import Union, Tuple
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


PADDING_FILL = -1.0


def load_and_reshape(
    file_name: str, max_seq_len: int, debug_data: Union[bool, int] = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from `file_name` and reshape into a 3D array of shape [num_examples, max_seq_len, num_features].
    A padding mask of data will also be produced (same shape), having elements True where time series were padded 
    (due to being shorter than max_seq_len).

    Args:
        file_name (str): Original data CSV file.
        max_seq_len (int): The maximum allowed length of the time-series dimension of the reshaped data.
        debug_data (Union[bool, int], optional): If int, read only top debug_data-many rows, if True, 
            read only top 10000 rows, if False read whole dataset. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [0] loaded and reshaped data, [1] corresponding padding.
    """
    padding_indicator = -999.0

    # Load data
    if debug_data is not False:
        if isinstance(debug_data, bool):
            nrows: Union[int, None] = 10000
        else:
            assert isinstance(debug_data, int), "debug_data argument must be bool or int."
            nrows = debug_data
    else:
        nrows = None
    ori_data = pd.read_csv(file_name, nrows=nrows)
    if ori_data.columns[0] == "Unnamed: 0":  # Remove spurious column, so that column 0 is now 'admissionid'.
        ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

    # Parameters
    uniq_id = np.unique(ori_data["admissionid"])
    no = len(uniq_id)
    dim = len(ori_data.columns) - 1

    # Output initialization
    assert np.any(ori_data == padding_indicator) == False, f"Padding indicator value {padding_indicator} found in data"
    loaded_data = np.empty([no, max_seq_len, dim])  # Shape: [no, max_seq_len, dim]
    loaded_data.fill(padding_indicator)

    # For each uniq id
    for i in tqdm(range(no)):

        # Extract the time-series data with a certain admissionid
        idx = ori_data.index[ori_data["admissionid"] == uniq_id[i]]
        curr_data = ori_data.iloc[idx].to_numpy()  # Shape: [curr_no, dim + 1]

        # Assign to the preprocessed data (Excluding ID)
        curr_no = len(curr_data)
        if curr_no >= max_seq_len:
            loaded_data[i, :, :] = curr_data[:max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
        else:
            loaded_data[i, -curr_no:, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]

    padding_mask = loaded_data == padding_indicator
    loaded_data = np.where(padding_mask, PADDING_FILL, loaded_data)

    return loaded_data, padding_mask


def to_3d(arr: np.ndarray, max_seq_len: int) -> np.ndarray:
    n_patients = arr.shape[0] // max_seq_len
    dim = arr.shape[1]
    return np.reshape(arr, [n_patients, max_seq_len, dim])


def to_2d(arr: np.ndarray) -> np.ndarray:
    n_patients = arr.shape[0]
    max_seq_len = arr.shape[1]
    dim = arr.shape[2]
    return np.reshape(arr, [n_patients * max_seq_len, dim])


def imputation(curr_data: np.ndarray, median_vals: np.ndarray, zero_fill: bool = True) -> np.ndarray:
    """Impute missing data using bfill, ffill and median imputation.

    Args:
        curr_data (np.ndarray): Data before imputation.
        median_vals (np.ndarray): Median values for each column.
        zero_fill (bool, optional): Whather to Fill with zeros the cases where median_val is nan. Defaults to True.

    Returns:
        np.ndarray: Imputed data.
    """

    curr_data = pd.DataFrame(data=curr_data)
    median_vals = pd.Series(median_vals)

    # Backward fill
    imputed_data = curr_data.bfill(axis="rows")
    # Forward fill
    imputed_data = imputed_data.ffill(axis="rows")
    # Median fill
    imputed_data = imputed_data.fillna(median_vals)

    # Zero-fill, in case the `median_vals` for a particular feature is `nan`.
    if zero_fill:
        imputed_data = imputed_data.fillna(0.0)

    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()


def get_medians(data: np.ndarray, padding_mask: np.ndarray):
    assert len(data.shape) == 3

    max_seq_len = data.shape[1]
    data = to_2d(data)
    if padding_mask is not None:
        padding_mask = to_2d(padding_mask)
        data_temp = np.where(padding_mask, np.nan, data)  # To avoid PADDING_INDICATOR affecting results.
    else:
        data_temp = data

    # Medians
    median_vals = np.nanmedian(data_temp, axis=0)  # Shape: [dim + 1]

    return median_vals


def get_scaler(data: np.ndarray, padding_mask: np.ndarray):
    assert len(data.shape) == 3

    max_seq_len = data.shape[1]
    data = to_2d(data)
    if padding_mask is not None:
        padding_mask = to_2d(padding_mask)
        data_temp = np.where(padding_mask, np.nan, data)  # To avoid PADDING_INDICATOR affecting results.
    else:
        data_temp = data

    # Scaler
    scaler = MinMaxScaler()
    scaler.fit(data_temp)  # Note that np.nan's will be left untouched.

    return scaler


def impute(data: np.ndarray, padding_mask: np.ndarray, median_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    assert len(data.shape) == 3

    data_imputed_ = np.zeros_like(data)

    for i in range(data.shape[0]):
        cur_data = data[i, :, :]
        if padding_mask is not None:
            cur_data = np.where(padding_mask[i, :, :], np.nan, cur_data)

        # Scale and impute (excluding time)
        cur_data_imputed = imputation(cur_data, median_vals)  # V

        # Update
        data_imputed_[i, :, :] = cur_data_imputed

    # Set padding
    if padding_mask is not None:
        data_imputed_ = np.where(padding_mask, PADDING_FILL, data_imputed_)

    return data_imputed_


def process(data: np.ndarray, padding_mask: np.ndarray, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]:

    assert len(data.shape) == 3

    data_ = np.zeros_like(data)

    for i in range(data.shape[0]):
        cur_data = data[i, :, :]
        if padding_mask is not None:
            cur_data = np.where(padding_mask[i, :, :], np.nan, cur_data)

        # Preprocess time (0th element of dim. 2):
        preprocessed_time = cur_data[:, 0] - np.nanmin(cur_data[:, 0])

        # Scale and impute (excluding time)
        cur_data = scaler.transform(cur_data)

        # Set time
        cur_data[:, 0] = preprocessed_time

        # Update
        data_[i, :, :] = cur_data

    # Set padding
    if padding_mask is not None:
        data_ = np.where(padding_mask, PADDING_FILL, data_)

    return data_


def preprocess_data(data: np.ndarray, padding_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess and impute `data`.

    Args:
        data (np.ndarray of float): 
            Data as loaded (and reshaped to 3D). Shape [num_examples, max_seq_len, num_features].
        padding_mask (np.ndarray of bool): 
            Padding mask of data, indicating True where time series were shorter than max_seq_len and were padded. 
            Same shape as data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [0] preprocessed data, [1] preprocessed and imputed data.
    """
    median_vals = get_medians(data, padding_mask)
    imputed_data = impute(data, padding_mask, median_vals)

    scaler_i = get_scaler(imputed_data, padding_mask)
    imputed_processed_data = process(imputed_data, padding_mask, scaler_i)

    scaler_o = get_scaler(data, padding_mask)
    processed_data = process(data, padding_mask, scaler_o)

    return processed_data, imputed_processed_data
