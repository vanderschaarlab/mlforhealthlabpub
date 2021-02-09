"""
Utilities related to solutions parsing, IO, validation etc.
"""
import os
from typing import Union, Tuple, List

import numpy as np

# Resolve data package.
try:
    import common.data.data_preprocess as prp
    from common.data.data_utils import data_division
except ModuleNotFoundError:
    try:
        import data.data_preprocess as prp  # type: ignore
        from data.data_utils import data_division  # type: ignore
    except ModuleNotFoundError:
        import utils.data_preprocess as prp  # type: ignore
        from utils.data_utils import data_division  # type: ignore


def load_data(
    data_dir: str,
    data_file_name: str,
    max_seq_len: int,
    seed: int,
    train_rate: float,
    force_reprocess: bool,
    debug_data: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and reshape the input (Amsterdam) data, get train/test splits.

    Args:
        data_dir (str): Directory where the data file is located.
        data_file_name (str): Data CSV file name (needs to end in ".csv").
        max_seq_len (int): Maximum sequence length of the time series dimension - for reshaping.
        seed (int): Random seed for data split.
        train_rate (float): The fraction of the data to allocate to training set.
        force_reprocess (bool): If True, will always preprocess the data rather than trying to load preprocessed data 
            if available.
        debug_data (Union[int, bool], optional): If int, read only top debug_data-many rows, if True, 
            read only top 10000 rows, if False read whole dataset. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: [description]
    """

    npz_path = os.path.join(data_dir, data_file_name.replace(".csv", ".npz"))

    if os.path.exists(npz_path) and not force_reprocess:

        with np.load(npz_path) as data:
            raw_data = data["raw_data"]
            padding_mask = data["padding_mask"]
            train_idx = data["train_idx"]
            test_idx = data["test_idx"]

    else:

        data_path = os.path.join(data_dir, data_file_name)
        raw_data, padding_mask = prp.load_and_reshape(data_path, max_seq_len, debug_data)
        _, (train_idx, test_idx) = data_division(raw_data, seed=seed, divide_rates=[train_rate, 1 - train_rate])

        np.savez(npz_path, raw_data=raw_data, padding_mask=padding_mask, train_idx=train_idx, test_idx=test_idx)

    return raw_data, padding_mask, train_idx, test_idx


def load_generated_data(data_path):
    """Load from `data_path` the generated data, and if available, the corresponding padding mask.

    Args:
        data_path (str): Path to the npz file containing generated data.

    Returns:
        generated_data (np.ndarray of float) [, generated_data_padding_mask (np.ndarray of bool)]
    """
    with np.load(data_path) as data:
        generated_data = data["generated_data"]
        if np.shape(data["padding_mask"]) != (0,):  # Check if `padding_mask` provided.
            generated_data_padding_mask = data["padding_mask"]
        else:
            generated_data_padding_mask = None
    return generated_data, generated_data_padding_mask


def parse_hider_output(hider_output: Union[Tuple, np.ndarray]):
    """Checks that the output of the hider function is as expected and parse it to return:
    generated_data (np.ndarray of float), generated_data_padding_mask ((np.ndarray of bool) OR None)
    """
    exc_message = (
        "Hider output must be like: \n"
        "(a) generated_data: np.ndarray or \n"
        "(b) (generated_data: np.ndarray, eval_seeds: int / None) or \n"
        "(c) (generated_data: np.ndarray, padding_mask: np.ndarray / None) or \n"
        "(d) (generated_data: np.ndarray, padding_mask: np.ndarray / None, eval_seeds: int / None)."
    )
    if isinstance(hider_output, (tuple, np.ndarray)):
        if isinstance(hider_output, np.ndarray):
            return hider_output, None, None
        else:
            if (
                len(hider_output) == 2
                and isinstance(hider_output[0], np.ndarray)
                and isinstance(hider_output[1], (np.ndarray, type(None), int))
            ):
                if isinstance(hider_output[1], int):
                    return hider_output[0], None, hider_output[1]
                else:
                    return hider_output[0], hider_output[1], None
            elif (
                len(hider_output) == 3
                and isinstance(hider_output[0], np.ndarray)
                and isinstance(hider_output[1], (np.ndarray, type(None)))
                and isinstance(hider_output[2], (int, type(None)))
            ):
                return hider_output[0], hider_output[1], hider_output[2]
            else:
                raise RuntimeError(exc_message)
    else:
        raise RuntimeError(exc_message)


def validate_hider_output(
    hider: str, hider_dir: str, features: list, data_shape: tuple, raise_exception=True, skip_fp=False, skip_osa=False
):
    """Perform a number of checks on the hider output.

    Args:
        hider (str): Name of hider.
        hider_dir (str): Directory of hider output.
        features (list): List of feature indices for feature prediction evaluation.
        data_shape (tuple): Shape of hider **input** data.
        raise_exception (bool, optional): Whether to raise exception on validation failure (else will just 
            return False). Defaults to True.
        skip_fp (bool, optional): Skip feature prediction output presence check. Defaults to False.
        skip_osa (bool, optional): Skip one-step-ahead prediction output presence check. Defaults to False.

    Raises:
        RuntimeError: If raise_exception is True and validation failed.

    Returns:
        bool: True if validation passed, False if validation failed (when raise_exception is False).
    """

    data_path = os.path.join(hider_dir, "data.npz")
    feature_scores_path = os.path.join(hider_dir, "feature_prediction_scores.txt")
    osa_path = os.path.join(hider_dir, "osa_score.txt")

    try:
        generated_data, generated_data_padding_mask = load_generated_data(data_path)
    except (IOError, ValueError) as ex:
        if raise_exception:
            raise RuntimeError(  # pylint: disable=raise-missing-from
                "Problem loading hider generated data. "
                f"It is possible hider solution '{hider}' encountered an error."
            )
        else:
            return False
    if generated_data.shape != data_shape:
        if raise_exception:
            raise RuntimeError(
                f"Hider '{hider}' generated data has a shape mismatch with input data. "
                f"Expected generated data shape (= input data shape): {data_shape}, "
                f"but got shape {generated_data.shape}."
            )
        else:
            return False
    if generated_data_padding_mask is None:
        pass  # Padding mask not provided - OK.
    else:
        if generated_data.shape != generated_data_padding_mask.shape:
            if raise_exception:
                raise RuntimeError(
                    f"Hider '{hider}' generated data has a shape mismatch with generated data padding mask. "
                    f"Expected generated data padding mask shape (= generated data shape): {generated_data.shape}, "
                    f"but got shape {generated_data_padding_mask.shape}."
                )
            else:
                return False

    if not skip_fp:
        if os.path.isfile(feature_scores_path):
            features_read = []
            with open(feature_scores_path) as f:
                for line in f:
                    features_read.append(float(line))
            if len(features_read) != len(features):
                if raise_exception:
                    raise RuntimeError(
                        f"Feature prediction scores could not be evaluated for all test features. "
                        f"It is possible hider solution '{hider}' encountered an error."
                    )
                else:
                    return False
            else:
                pass  # Feature predictions OK.
        else:
            if raise_exception:
                raise RuntimeError(
                    f"Feature prediction scores result not found. "
                    f"It is possible hider solution '{hider}' encountered an error."
                )
            else:
                return False
    else:
        pass

    if not skip_osa:
        if os.path.isfile(osa_path):
            pass  # OSA OK.
        else:
            if raise_exception:
                raise RuntimeError(
                    f"One-step-ahead prediction result not found. "
                    f"It is possible hider solution '{hider}' encountered an error."
                )
            else:
                return False
    else:
        pass

    return True  # All validation passed.


def validate_seeker_output(seeker: str, seeker_output_path: str, labels: np.ndarray, raise_exception=True):
    """Perform a number of checks on the seeker output.

    Args:
        seeker (str): Name of seeker.
        seeker_output_path (str): Directory of seeker output.
        labels (np.ndarray): True reidentification labels.
        raise_exception (bool, optional): Whether to raise exception on validation failure (else will just 
            return False). Defaults to True. Defaults to True.

    Raises:
        RuntimeError: If raise_exception is True and validation failed.

    Returns:
        Returns reidentified_data (seeker output) if validation passed.
        Returns False if validation failed (when raise_exception is False).
    """
    if os.path.isfile(seeker_output_path):
        try:
            with np.load(seeker_output_path) as data:
                reidentified_data = data["reidentified_data"]
        except (IOError, ValueError) as ex:
            if raise_exception:
                raise RuntimeError(  # pylint: disable=raise-missing-from
                    "Problem loading hider-vs-seeker evaluation results. "
                    f"It is possible seeker solution '{seeker}' has invalid output format."
                )
            else:
                return False
        len_ = labels.shape[0]
        if reidentified_data.shape in [(len_,), (len_, 1), (1, len_)]:
            reidentified_data = np.reshape(reidentified_data, (len_,))
            return reidentified_data
        else:
            raise RuntimeError(
                f"Seeker solution '{seeker}' has unexpected output shape: {reidentified_data.shape}. "
                f"Should be like: ({len_},)."
            )
    else:
        if raise_exception:
            raise RuntimeError(
                "Couldn't find hider-vs-seeker evaluation results. "
                f"It is possible seeker solution '{seeker}' encountered an error."
            )
        else:
            return False


def _print_feat_scores(scores: List[float], task_types: List[str]) -> None:
    txt = ""
    for idx, score in enumerate(scores):
        task_type = task_types[idx]
        if idx > 0:
            txt += ",\t"
        txt += f"{score:.5g} ({'AUC' if task_type == 'classification' else 'RMSE'})"
    print(txt)


def benchmark_hider(
    feat_scores: List[float],
    task_types: List[str],
    osa_score: float,
    eval_feat_scores: List[float],
    eval_task_types: List[str],
    eval_osa_score: float,
    threshold_auroc: float,
    threshold_rmse: float,
) -> bool:
    """Benchmark the results of a hider solution based on feature prediction and one-step-ahead prediction evaluation. 

    Args:
        feat_scores (List[float]): Solution feature prediction scores list.
        task_types (List[str]): Solution feature task types ('classification', 'regression') list.
        osa_score (float): Solution one-step-ahead prediction score.
        eval_feat_scores (List[float]): Evaluation feature prediction scores list.
        eval_task_types (List[str]): Evaluation feature task types ('classification', 'regression') list.
        eval_osa_score (float): Evaluation one-step-ahead prediction score.
        threshold_auroc (float): Evaluation threshold, AUROC (classification case).
        threshold_rmse (float): Evaluation threshold, RMSE (regression case).

    Returns:
        bool: Whether benchmark was passed.
    """

    failed = False

    print(f"\nThreshold regression = {threshold_rmse}\nThreshold classification = {threshold_auroc}")

    # Feature prediction.
    print("\n[Feature prediction]")
    print("Benchmark feature prediction scores:")
    _print_feat_scores(eval_feat_scores, eval_task_types)
    print("Solution feature prediction scores:")
    _print_feat_scores(feat_scores, task_types)
    for idx, eval_score in enumerate(eval_feat_scores):
        eval_type = eval_task_types[idx]
        soln_score = feat_scores[idx]
        soln_type = task_types[idx]
        if eval_type != soln_type:
            print(f"Feature prediction: evaluation feature {idx + 1}. Expected: '{eval_type}'. Got: '{soln_type}'.")
            failed = True
        if eval_type == "classification":
            if soln_score < threshold_auroc * eval_score:
                print(
                    "Feature prediction: "
                    f"evaluation feature {idx + 1}. {soln_score} < {threshold_auroc} * {eval_score}."
                )
                failed = True
        else:
            if soln_score > threshold_rmse * eval_score:
                print(
                    "Feature prediction: "
                    f"evaluation feature {idx + 1}. {soln_score} > {threshold_rmse} * {eval_score}."
                )
                failed = True

    # OSA.
    print("\n[One-step-ahead prediction]")
    print(f"Benchmark One-step-ahead prediction score:\t{eval_osa_score}")
    print(f"Solution One-step-ahead prediction score:\t{osa_score}")
    if osa_score > threshold_rmse * eval_osa_score:
        print(f"One-step-ahead prediction score: {osa_score} > {threshold_rmse} * {eval_osa_score}.")
        failed = True
    print()

    return not failed
