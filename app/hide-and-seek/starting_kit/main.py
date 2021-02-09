"""
Script for competitors to locally test their solutions.

The script follows the logic of the competition's scoring process. The hider from `hider.py` and the seeker from 
`seeker.py` will be imported and played against each other. Should be executed from containing directory. 
See the command help:
```sh
$ python main.py --help
```
See also docstring of `main()` for more details.

Note:
    The script requires the dependencies listed in `requirements.txt`. It can also be ran without tensorflow==1.15.2
    or keras==2.3.1, but in that case, some parts of the script will be skipped.

Last updated Date: Oct 17th 2020
Code author: Evgeny Saveliev
Contact: e.s.saveliev@gmail.com
"""
import os
import argparse
import shutil

import numpy as np

from utils.misc import (
    tf115_found,
    tfdeterminism_found,
    fix_all_random_seeds,
    temp_seed_numpy,
    in_progress,
    tf_fixed_seed_seesion,
)

if tf115_found:
    from utils.misc import tf_set_log_level
    import logging

    tf_set_log_level(logging.FATAL)

    # # May be useful for determinism:
    # import tensorflow as tf
    # from keras import backend as K
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)

    if tfdeterminism_found:
        from tfdeterminism import patch

        patch()

else:
    print("Warning: TensorFlow 1.15 was not found so the parts of the script that rely on it will be skipped.")

import utils.data_preprocess as prp
from utils.solutions import (
    load_data,
    load_generated_data,
    parse_hider_output,
    validate_hider_output,
    validate_seeker_output,
    benchmark_hider,
)
from utils.metric_utils import reidentify_score, feature_prediction, one_step_ahead_prediction

import hider as hider_module
import seeker as seeker_module


def main(args):
    """The main script - hider from `hider.py` and the seeker from `seeker.py` will be imported and played against 
    each other.

    Stages of the script:
        * Load data.
        * Run the hider.
        * Evaluate hider via feature prediction and one-step-ahead prediction.
        * Run the seeker (on the hider's generated data).

    Args:
        args (argparse.Namespace): parsed arguments from the command line.

    Raises:
        ValueError: in case there are issues with required files or directories.
    """

    # ================================================= System setup. ==================================================

    # If no TensorFlow 1.15 found on the system, skip parts of the script.
    if not tf115_found:
        args.skip_fp = True
        args.skip_osa = True

    # Fix random seeds.
    fix_all_random_seeds(args.seed)
    # NOTE:
    # The fix_all_random_seeds() call may not be sufficient to make tensorflow fully deterministic.
    # See, for example: https://github.com/NVIDIA/framework-determinism

    # ============================================== Prepare directories. ==============================================
    # Code directory.
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}.")
    print(f"\nCode directory:\t\t{code_dir}")

    # Data path.
    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at {data_path}.")
    print(f"Data file:\t\t{data_path}")
    data_dir = os.path.dirname(data_path)
    data_file_name = os.path.basename(data_path)

    # Output directories.
    out_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory:\t{out_dir}")
    hider_dir = os.path.join(out_dir, "hider")
    if os.path.exists(hider_dir):
        shutil.rmtree(hider_dir)
    os.makedirs(hider_dir, exist_ok=True)
    seeker_dir = os.path.join(out_dir, "seeker")
    if os.path.exists(seeker_dir):
        shutil.rmtree(seeker_dir)
    os.makedirs(seeker_dir, exist_ok=True)
    print(f"  ├ Hider output:\t{hider_dir}")
    print(f"  └ Seeker output:\t{seeker_dir}\n")

    # =================================================== Load data. ===================================================
    if args.debug_data <= 0:
        args.debug_data = False
    with in_progress("Preprocessing and loading data"):
        original_data, original_padding_mask, train_idx, test_idx = load_data(
            data_dir=data_dir,
            data_file_name=data_file_name,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            train_rate=args.train_frac,
            force_reprocess=True,  # If True, re-preprocess data every time (rather than reusing).
            debug_data=args.debug_data,
        )
    print(f"\nOriginal data preview (original_data[:2, -10:, :2]):\n{original_data[:2, -10:, :2]}\n")

    # ================================================= Part I: Hider. =================================================
    # Set up hider input.
    original_data_train = original_data[train_idx]
    original_padding_mask_train = original_padding_mask[train_idx]
    hider_input = {"data": original_data_train, "seed": args.seed, "padding_mask": original_padding_mask_train}

    # Run hider.
    with in_progress("Running Hider"):
        hider_output = hider_module.hider(hider_input)
        generated_data, generated_data_padding_mask, _ = parse_hider_output(hider_output)
    print(f"\nGenerated data preview (generated_data[:2, -10:, :2]):\n{generated_data[:2, -10:, :2]}\n")

    # Save hider output.
    hider_output_file = os.path.join(hider_dir, "data.npz")
    np.savez(
        hider_output_file,
        generated_data=generated_data,
        padding_mask=generated_data_padding_mask if generated_data_padding_mask is not None else [],
    )

    # Evaluate hider.

    # - Prepare data
    if not (args.skip_fp and args.skip_osa):
        with in_progress("Preparing data for hider evaluation"):
            generated_data, generated_data_padding_mask = load_generated_data(hider_output_file)
            _, original_data_train_imputed = prp.preprocess_data(original_data_train, original_padding_mask_train)
            _, generated_data_imputed = prp.preprocess_data(generated_data, generated_data_padding_mask)
            _, original_data_test_imputed = prp.preprocess_data(
                original_data[test_idx], original_padding_mask[test_idx]
            )

    # - Feature prediction step.
    if not args.skip_fp:
        num_features = original_data_train.shape[2]
        with temp_seed_numpy(args.seed):
            feature_idx = np.random.permutation(num_features)[: args.feature_prediction_no]
        print(f"\nFeature prediction evaluation on IDs: {feature_idx}\n")

        with in_progress("Running feature prediction"):
            with in_progress("Running on [original data]"):
                with tf_fixed_seed_seesion(args.seed):
                    original_feature_prediction_accuracy, ori_task_types = feature_prediction(
                        train_data=original_data_train_imputed,
                        test_data=original_data_test_imputed,
                        index=feature_idx,
                        verbose=args.eval_verbose,
                    )
            with in_progress("Running on [generated data]"):
                with tf_fixed_seed_seesion(args.seed):
                    new_feature_prediction_accuracy, new_task_types = feature_prediction(
                        train_data=generated_data_imputed,
                        test_data=original_data_test_imputed,
                        index=feature_idx,
                        verbose=args.eval_verbose,
                    )

        print("\nFeature prediction errors (per feature):")
        print(f"Original data:\t\t{original_feature_prediction_accuracy}")
        print(f"New (hider-generated):\t{new_feature_prediction_accuracy}\n")

        # - Save results.
        with open(os.path.join(hider_dir, "feature_prediction_scores.txt"), "w") as f:
            for score in new_feature_prediction_accuracy:
                print(score.astype(str), file=f)

    else:
        print(f"Feature prediction step skipped!{ '' if tf115_found else ' (TensorFlow 1.15 not found)' }\n")

    # - One-step-ahead prediction step.
    if not args.skip_osa:
        with in_progress("Running one-step-ahead prediction"):
            with in_progress("Running on [original data]"):
                with tf_fixed_seed_seesion(args.seed):
                    original_osa_perf = one_step_ahead_prediction(
                        train_data=original_data_train_imputed,
                        test_data=original_data_test_imputed,
                        verbose=args.eval_verbose,
                    )
            with in_progress("Running on [generated data]"):
                with tf_fixed_seed_seesion(args.seed):
                    new_osa_perf = one_step_ahead_prediction(
                        train_data=generated_data_imputed,
                        test_data=original_data_test_imputed,
                        verbose=args.eval_verbose,
                    )

        print("\nOne-step-ahead prediction errors (per feature):")
        print(f"Original data:\t\t{original_osa_perf}")
        print(f"New (hider-generated):\t{new_osa_perf}\n")

        # - Save results.
        with open(os.path.join(hider_dir, "osa_score.txt"), "w") as f:
            print(new_osa_perf.astype(str), file=f)

    else:
        print(f"One-step-ahead prediction step skipped!{ '' if tf115_found else ' (TensorFlow 1.15 not found)' }\n")

    if not args.skip_fp and not args.skip_osa:
        passed = benchmark_hider(
            feat_scores=new_feature_prediction_accuracy,
            task_types=new_task_types,
            osa_score=new_osa_perf,
            eval_feat_scores=original_feature_prediction_accuracy,
            eval_task_types=ori_task_types,
            eval_osa_score=original_osa_perf,
            threshold_auroc=0.85,
            threshold_rmse=5.00,
        )
        print(f'>>> Hider evaluation: {"passed" if passed else "failed"}')

    # Validation of hider results:
    validate_hider_output(
        hider="hider from hider.py",
        hider_dir=hider_dir,
        features=feature_idx if not args.skip_fp else None,
        data_shape=original_data_train.shape,
        raise_exception=True,
        skip_fp=args.skip_fp,
        skip_osa=args.skip_osa,
    )

    # ======================================= Part II: Seeker (vs Part I Hider). =======================================
    # Set up seeker input.
    seeker_input = {
        "generated_data": generated_data,
        "enlarged_data": original_data,
        "seed": args.seed,
        "generated_data_padding_mask": generated_data_padding_mask,
        "enlarged_data_padding_mask": original_padding_mask,
    }

    # Run seeker.
    with in_progress("Running Seeker"):
        reidentified_labels = seeker_module.seeker(seeker_input)

    # Save seeker output.
    seeker_output_file = os.path.join(seeker_dir, "data.npz")
    np.savez(seeker_output_file, reidentified_data=reidentified_labels)

    # Evaluate seeker (vs hider).
    true_labels = np.isin(np.arange(original_data.shape[0]), train_idx)
    reidentified_labels = validate_seeker_output(
        seeker="seeker from seeker.py", seeker_output_path=seeker_output_file, labels=true_labels, raise_exception=True
    )
    reidentification_score = reidentify_score(true_labels, reidentified_labels)

    print(f"\nTrue labels:\t\t\t\t{true_labels.astype(int)}")
    print(f"Reidentified (by seeker) labels:\t{reidentified_labels}")
    print(f"Reidentification score:\t\t\t{reidentification_score:.4f}\n")


if __name__ == "__main__":

    # Inputs for the main function
    parser = argparse.ArgumentParser(
        description="A script that emulates the competition's scoring process, "
        "the hider (from hider.py) is run against the seeker (from seeker.py)."
    )
    parser.add_argument(
        "-d",
        "--data_path",
        metavar="PATH",
        default="./data/train_longitudinal_data.csv",
        type=str,
        help="Data file path (Amsterdam dataset). Defaults to './data/train_longitudinal_data.csv'.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar="PATH",
        default="./output",
        type=str,
        help="Output directory. Defaults to './output'.",
    )
    parser.add_argument(
        "-m", "--max_seq_len", metavar="INT", default=100, type=int, help="Max sequence length limit. Defaults to 100."
    )
    parser.add_argument(
        "-t", "--train_frac", default=0.5, metavar="FLOAT", type=float, help="Training set fraction. Defaults to 0.5."
    )
    parser.add_argument(
        "-e",
        "--hider_eval_threshold",
        default=0.85,
        metavar="FLOAT",
        type=float,
        help="Hider evaluation threshold. Defaults to 0.85.",
    )
    parser.add_argument(
        "-f",
        "--feature_prediction_no",
        metavar="INT",
        default=5,
        type=int,
        help="Number of features in the subset of features used to run feature prediction "
        "(part of hider evaluation). Defaults to 5.",
    )
    parser.add_argument("-s", "--seed", metavar="INT", default=0, type=int, help="Random seed. Defaults to 0.")
    parser.add_argument(
        "-g",
        "--debug_data",
        metavar="INT",
        default=0,
        type=int,
        help="Set this to a non-0 value to use a 'debug' subset of the dataset instead of the whole dataset "
        "(useful for speedy debugging), only the first --debug_data many rows of the data file will be loaded. "
        "Defaults to 0.",
    )
    parser.add_argument(
        "--skip_fp", action="store_true", default=False, help="Skip feature prediction step of hider evaluation if set."
    )
    parser.add_argument(
        "--skip_osa",
        action="store_true",
        default=False,
        help="Skip one-step-ahead prediction step of hider evaluation if set.",
    )
    parser.add_argument(
        "--eval_verbose",
        action="store_true",
        default=False,
        help="If set, the underlying training in hider evaluation stages will be shown verbosely "
        "(training epoch etc.).",
    )

    parsed_args = parser.parse_args()

    # Call main function
    main(parsed_args)
