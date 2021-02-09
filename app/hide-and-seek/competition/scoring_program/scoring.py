#!/usr/bin/env python3

import argparse
import docker
import os
import shutil
import sys
import ast
import socket

import numpy as np
from tfdeterminism import patch

patch()  # Ensure tf GPU determinism: https://github.com/NVIDIA/framework-determinism

import data.data_preprocess as prp
from metrics.metric_utils import feature_prediction, one_step_ahead_prediction, reidentify_score
from computils.misc import fix_all_random_seeds, tf_fixed_seed_seesion, redact_exception
from computils.solutions import (
    load_data,
    load_generated_data,
    validate_hider_output,
    validate_seeker_output,
    benchmark_hider,
)
from running import competition_config, MAX_SEQ_LEN, TRAIN_RATE, DATA_FILE_NAME, FORCE_REPROCESS, SEED, DEBUG_DATA

DEFAULT_IMAGE = competition_config.get("DEFAULT_IMAGE", "drshushen/hide-and-seek-codalab:latest")
HIDER_EVAL_TRAINING_VERBOSE = competition_config.get("HIDER_EVAL_TRAINING_VERBOSE", False)
HIDER_EVAL_TRAINING_DEBUG = competition_config.get("HIDER_EVAL_TRAINING_DEBUG", False)
HIDER_EVAL_FEATURES_DEBUG = competition_config.get("HIDER_EVAL_FEATURES_DEBUG", False)
HIDER_EVAL_FORCE_BENCHMARK_RECALC = competition_config.get("HIDER_EVAL_FORCE_BENCHMARK_RECALC", False)
HIDER_EVAL_RERUN_SOLUTION_EVERY_SEED = competition_config.get("HIDER_EVAL_RERUN_SOLUTION_EVERY_SEED", False)
HIDER_EVAL_RAISE_EXCEPTION = competition_config.get("HIDER_EVAL_RAISE_EXCEPTION", False)
FEATURES_EVAL = competition_config["FEATURES_EVAL"]

fix_all_random_seeds(SEED)
SEED_FOR_SEEKERS = competition_config.get("SEED_FOR_SEEKERS", 12345)
SEEDS_FOR_HIDERS = competition_config.get("SEEDS_FOR_HIDERS", (12345, 42, 1, 123, 777))

SEEKER_SCORE_NA = competition_config.get("SEEKER_SCORE_NA", -9.99)
HIDER_SCORE_NA = competition_config.get("HIDER_SCORE_NA", 9.99)
HIDER_BENCHMARK_THRESHOLD_AUROC = competition_config.get("HIDER_BENCHMARK_THRESHOLD_AUROC", 0.85)
HIDER_BENCHMARK_THRESHOLD_RMSE = competition_config.get("HIDER_BENCHMARK_THRESHOLD_RMSE", 5.00)
COMPETITION_STAGE_EVALUATION = competition_config.get("COMPETITION_STAGE_EVALUATION", False)

HIDERS_EVAL_DIRNAME = "hiders.eval"

def _to_str(o):
    if isinstance(o, np.ndarray):
        return o.astype(str)
    else:
        return str(o)


def _docker_image(code_dir):
    image_path = os.path.join(code_dir, "Dockerimage")
    if os.path.exists(image_path):
        with open(image_path, "r") as f:
            return f.read().strip()
    else:
        return DEFAULT_IMAGE


def _docker_pull(client, image):
    split = image.split(":")
    if len(split) > 1:
        image, tag = split
    else:
        tag = "latest"
    client.images.pull(image, tag)


def _run_container_with_managed_logging(container, logs_dir, quiet_flag):
    hider_stdout = os.path.join(logs_dir, "stdout")
    hider_stderr = os.path.join(logs_dir, "stderr")
    if os.path.exists(hider_stdout):
        os.remove(hider_stdout)
    if os.path.exists(hider_stderr):
        os.remove(hider_stderr)
    try:
        if not quiet_flag:
            for log in container.logs(stream=True, follow=True):
                sys.stdout.buffer.write(log)
        else:
            for log in container.logs(stream=True, follow=True, stdout=True, stderr=False):
                with open(hider_stdout, "ab") as f:
                    f.write(log)
            for log in container.logs(stream=True, follow=True, stdout=False, stderr=True):
                with open(hider_stderr, "ab") as f:
                    f.write(log)
        container.wait()
    finally:
        container.stop()
        container.remove(force=True)


def _load_dumped_exc(dump_path):
    if os.path.exists(dump_path):
        with open(dump_path, "r") as f:
            dumped_exc = "".join(f.readlines())
        return dumped_exc
    else:
        return False


def _process_msg_to_scorer(hider_dir, n_eval_seeds_max):
    # Returns num_eval_seeds (int)
    msg_to_scorer = os.path.join(hider_dir, "MSG")
    if os.path.exists(msg_to_scorer):
        with open(msg_to_scorer, "r") as f:
            msg = f.readline()
        if msg == "rescore":
            return 0  # i.e. skip evaluation.
        else:
            is_eval_seeds = False
            try:
                eval_seeds = int(msg)
                is_eval_seeds = True
                if eval_seeds < 0:
                    eval_seeds = 0
                elif eval_seeds > n_eval_seeds_max:
                    eval_seeds = n_eval_seeds_max
                return eval_seeds
            except ValueError:
                pass
            if not is_eval_seeds:
                os.remove(msg_to_scorer)
    return n_eval_seeds_max


def _collapse_score_grids(feat_scores_grid, osa_scores_grid):
    feat_scores_mean = np.mean(feat_scores_grid, axis=0)
    feat_scores_std = np.std(feat_scores_grid, axis=0)
    osa_scores_mean = np.mean(osa_scores_grid, axis=0).item()
    osa_scores_std = np.std(osa_scores_grid, axis=0).item()
    return feat_scores_mean, osa_scores_mean, feat_scores_std, osa_scores_std


def _print_scores(
    feat_scores_grid, osa_scores_grid, feat_scores_mean, osa_scores_mean, feat_scores_std, osa_scores_std
):
    print(f"\nFeature prediction scores (n_seeds x feature_scores):\n{feat_scores_grid}")
    print(f"One-step-ahead prediction scores (n_seeds x 1):\n{osa_scores_grid}")
    feat_str = [f"{m:.5f}±{feat_scores_std[idx]:.5f}" for idx, m in enumerate(feat_scores_mean)]
    osa_str = f"{osa_scores_mean:.3f}±{osa_scores_std:.3f}"
    print(f"Feature prediction scores summarised:\n{feat_str}")
    print(f"One-step-ahead prediction scores summarised:\n{osa_str}\n")


def _get_hider_eval_benchmarks(eval_dir, raw_data, raw_data_padding_mask, train_idx, test_idx, features):

    # Hider evaluation benchmark.
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    eval_feat_pred = os.path.join(eval_dir, "feature_prediction_scores.txt")
    eval_feat_pred_types = os.path.join(eval_dir, "feature_prediction_scores.task_types.txt")
    eval_osa_pred = os.path.join(eval_dir, "osa_score.txt")

    if (
        not (os.path.exists(eval_feat_pred) and os.path.exists(eval_feat_pred_types) and os.path.exists(eval_osa_pred))
        or HIDER_EVAL_FORCE_BENCHMARK_RECALC
    ):
        # Obtain scoring results as a mean of reps.
        n_seeds = len(SEEDS_FOR_HIDERS)
        feat_scores_grid = np.zeros([n_seeds, len(features)], dtype=float)
        osa_scores_grid = np.zeros([n_seeds, 1], dtype=float)

        # Need to one-off generate evaluational scores.
        print("\nCalculating scores for hider evaluation benchmark...")
        _, train_data_imputed = prp.preprocess_data(raw_data[train_idx], raw_data_padding_mask[train_idx])
        _, test_data_imputed = prp.preprocess_data(raw_data[test_idx], raw_data_padding_mask[test_idx])

        for idx, seed in enumerate(SEEDS_FOR_HIDERS):

            print("\n" + "-" * 80)
            print(f"Seed {idx + 1}/{n_seeds}: {seed}")
            print("-" * 80 + "\n")

            # Feature prediction.
            with tf_fixed_seed_seesion(seed):
                feat_scores, task_types = feature_prediction(
                    train_data_imputed,
                    test_data_imputed,
                    features,
                    verbose=HIDER_EVAL_TRAINING_VERBOSE,
                    debug=HIDER_EVAL_TRAINING_DEBUG,
                )
            feat_scores_grid[idx, :] = feat_scores

            # One-step-ahead.
            with tf_fixed_seed_seesion(seed):
                osa_score = one_step_ahead_prediction(
                    train_data_imputed,
                    test_data_imputed,
                    verbose=HIDER_EVAL_TRAINING_VERBOSE,
                    debug=HIDER_EVAL_TRAINING_DEBUG,
                )
            osa_scores_grid[idx, :] = osa_score

        feat_scores_mean, osa_scores_mean, *stds = _collapse_score_grids(feat_scores_grid, osa_scores_grid)
        print("\nHider evaluation benchmark scores:")
        _print_scores(feat_scores_grid, osa_scores_grid, feat_scores_mean, osa_scores_mean, *stds)

        all_scores_path = os.path.join(eval_dir, "all_scores.npz")
        np.savez(all_scores_path, feat_scores_grid=feat_scores_grid, osa_scores_grid=osa_scores_grid)

        with open(eval_feat_pred, "w") as f:
            for score in feat_scores_mean:
                print(_to_str(score), file=f)
        with open(eval_feat_pred_types, "w") as f:
            for task_type in task_types:
                print(task_type, file=f)
        with open(eval_osa_pred, "w") as f:
            print(_to_str(osa_scores_mean), file=f)

    else:
        print("\nScores for hider evaluation benchmark found.")
        # Feature prediction.
        with open(eval_feat_pred, "r") as f:
            feat_scores = [float(x.strip()) for x in f.readlines()]
        with open(eval_feat_pred_types, "r") as f:
            task_types = [x.strip() for x in f.readlines()]
        # One-step-ahead.
        with open(eval_osa_pred, "r") as f:
            osa_score = float(f.read().strip())

    return feat_scores, task_types, osa_score


def process_vs_targets(targets_dir, vslist_arg):
    vslist_from_args = None
    if vslist_arg != "":
        try:
            vslist_from_args = ast.literal_eval(vslist_arg)
        except Exception:  # pylint: disable=broad-except
            print(f"Failed to parse --vslist arg: '{vslist_arg}'")
        if not isinstance(vslist_from_args, list):
            print("--vslist wasn't a list, ignoring")
    
    targets_list = [x for x in os.listdir(targets_dir) if x != HIDERS_EVAL_DIRNAME]
    if vslist_from_args is not None:
        targets_list = [x for x in targets_list if x in vslist_from_args]
    
    return targets_list


def _dockerize_vs(client, runtime, args, seeker, hider, is_seeker, seed):
    code_dir = os.path.join(args.opt_dir, "seekers", seeker, "res")
    vs_dir = os.path.join(args.opt_dir, "seekers", seeker, "vs", hider)
    exception_path = os.path.join(vs_dir, "EXCEPTION")
    self_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(vs_dir, exist_ok=True)

    hider_data = os.path.join(args.opt_dir, "hiders", hider, "data.npz")
    hider_mtime = os.path.getmtime(hider_data)
    score_file = os.path.join(vs_dir, "score.txt")
    if os.path.exists(score_file):
        if hider_mtime < os.path.getmtime(score_file) and not is_seeker:  # If is seeker always reevaluate.
            print("Re-using previously computed score for {} vs. {}".format(seeker, hider))
            with open(score_file, "r") as f:
                return float(f.read().strip())
        else:
            os.remove(score_file)

    if is_seeker:
        # Validate hider.
        data_dir = os.path.join(args.opt_dir, "data")
        raw_data, _, train_idx, _ = load_data(
            data_dir=data_dir,
            data_file_name=DATA_FILE_NAME,
            max_seq_len=MAX_SEQ_LEN,
            seed=SEED,
            train_rate=TRAIN_RATE,
            force_reprocess=FORCE_REPROCESS,
            debug_data=DEBUG_DATA,
        )
        features = [int(x) for x in FEATURES_EVAL]
        if HIDER_EVAL_FEATURES_DEBUG:
            features = features[:3]
        try:
            validate_hider_output(
                hider=hider,
                hider_dir=os.path.join(args.opt_dir, "hiders", hider),
                features=features,
                data_shape=raw_data[train_idx].shape,
                raise_exception=True,
            )
        except RuntimeError:
            print(f"\nSkipping vs hider '{hider}', as its output failed validation.\n")
            return None

    volumes = {
        args.opt_dir: {"bind": "/opt/hide-and-seek", "mode": "rw",},
        vs_dir: {"bind": os.path.join("/opt/hide-and-seek/seekers", seeker, "vs", hider), "mode": "rw",},
        self_dir: {"bind": "/opt/hide-and-seek/scoring", "mode": "ro",},
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    command = [
        "python3",
        "/opt/hide-and-seek/scoring/running.py",
        "--seeker",
        "--user",
        seeker,
        "--vs",
        hider,
        "--seed",
        str(seed),
        os.path.join("/opt/hide-and-seek/seekers", seeker),
        "/opt/hide-and-seek",
    ]

    image = _docker_image(code_dir)

    print(f"\nRunning {seeker} vs. {hider} in {image}...\n")
    container_launch_successful = False
    try:
        container = client.containers.run(
            image,
            command,
            runtime=runtime,
            detach=True,
            network="hide-and-seek",
            volumes=volumes,
            environment=environment,
        )
        container_launch_successful = True
    except Exception as ex:  # pylint: disable=broad-except
        print("Container launching failed.")
        msg = redact_exception(ex)
        if os.path.exists(exception_path):
            os.remove(exception_path)
        with open(exception_path, "w") as f:
            f.writelines(msg)
    if container_launch_successful:
        _run_container_with_managed_logging(container, vs_dir, args.quiet)

    dumped_exc = _load_dumped_exc(exception_path)
    if dumped_exc:
        if is_seeker:
            print(dumped_exc)
            raise RuntimeError("Exception encountered in running solution, see logs for details.")
        else:
            print(f"Exception encountered running seeker '{seeker}'")

    print("\nComputing reidentification score...")

    data_dir = os.path.join(args.opt_dir, "data")
    raw_data, padding_mask, train_idx, test_idx = load_data(
        data_dir=data_dir,
        data_file_name=DATA_FILE_NAME,
        max_seq_len=MAX_SEQ_LEN,
        seed=SEED,
        train_rate=TRAIN_RATE,
        force_reprocess=FORCE_REPROCESS,
        debug_data=DEBUG_DATA,
    )
    labels = np.isin(np.arange(raw_data.shape[0]), train_idx)

    seeker_output_path = os.path.join(vs_dir, "data.npz")
    reidentified_data = validate_seeker_output(seeker, seeker_output_path, labels, raise_exception=is_seeker)

    if reidentified_data is False:  # Reaching here also implies raise_seeker_ex is False.
        print(f"\nSkipping vs seeker '{seeker}', as its output failed validation.\n")
        reidentification_score = None
    else:
        reidentification_score = reidentify_score(labels, reidentified_data)
        with open(score_file, "w") as f:
            print(_to_str(reidentification_score), file=f)

    return reidentification_score


def _dockerize_hider(client, runtime, args):
    code_dir = os.path.join(args.input_dir, "res")
    hider_dir = os.path.join(args.opt_dir, "hiders", args.user)
    exception_path = os.path.join(hider_dir, "EXCEPTION")
    eval_dir = os.path.join(args.opt_dir, "hiders", HIDERS_EVAL_DIRNAME)
    self_dir = os.path.dirname(os.path.abspath(__file__))

    copy_dir = os.path.join(hider_dir, "res")
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    shutil.copytree(code_dir, copy_dir)

    os.makedirs(hider_dir, exist_ok=True)

    volumes = {
        args.opt_dir: {"bind": "/opt/hide-and-seek", "mode": "rw",},
        hider_dir: {"bind": os.path.join("/opt/hide-and-seek/hiders", args.user), "mode": "rw",},
        args.input_dir: {"bind": "/opt/hide-and-seek/input", "mode": "ro",},
        self_dir: {"bind": "/opt/hide-and-seek/scoring", "mode": "ro",},
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    image = _docker_image(code_dir)

    print("Pulling {}...".format(image))
    _docker_pull(client, image)

    print("Loading original data...")
    # Get raw data:
    data_dir = os.path.join(args.opt_dir, "data")
    raw_data, raw_data_padding_mask, train_idx, test_idx = load_data(
        data_dir=data_dir,
        data_file_name=DATA_FILE_NAME,
        max_seq_len=MAX_SEQ_LEN,
        seed=SEED,
        train_rate=TRAIN_RATE,
        force_reprocess=FORCE_REPROCESS,
        debug_data=DEBUG_DATA,
    )

    # Load testing features.
    features = [int(x) for x in FEATURES_EVAL]
    if HIDER_EVAL_FEATURES_DEBUG:
        features = features[:3]

    # Results grids.
    n_seeds = len(SEEDS_FOR_HIDERS)
    feat_scores_grid = np.zeros([n_seeds, len(features)], dtype=float)
    osa_scores_grid = np.zeros([n_seeds, 1], dtype=float)

    n_eval_seeds_actual = n_seeds
    skip_evaluation = False
    for idx, seed in enumerate(SEEDS_FOR_HIDERS):

        if idx == 0 or HIDER_EVAL_RERUN_SOLUTION_EVERY_SEED:
            command = [
                "python3",
                "/opt/hide-and-seek/scoring/running.py",
                "--hider",
                "--user",
                args.user,
                "--seed",
                str(seed),
                "/opt/hide-and-seek/input",
                "/opt/hide-and-seek",
            ]

            print(f"\nRunning {args.user} in {image}...\n")
            container_launch_successful = False
            try:
                container = client.containers.run(
                    image,
                    command,
                    runtime=runtime,
                    detach=True,
                    network="hide-and-seek",
                    volumes=volumes,
                    environment=environment,
                )
                container_launch_successful = True
            except Exception as ex:  # pylint: disable=broad-except
                print("Container launching failed.")
                msg = redact_exception(ex)
                if os.path.exists(exception_path):
                    os.remove(exception_path)
                with open(exception_path, "w") as f:
                    f.writelines(msg)
            if container_launch_successful:
                _run_container_with_managed_logging(container, hider_dir, args.quiet)

            dumped_exc = _load_dumped_exc(exception_path)
            if dumped_exc:
                print(dumped_exc)
                raise RuntimeError("Exception encountered in running solution, see logs for details.")
            
            if idx == 0:
                # NOTE: This processing only happens once, initially.
                
                # Check if submission requested to run on fewer than max number eval. seeds.
                n_eval_seeds_actual = _process_msg_to_scorer(hider_dir, n_seeds)
                if COMPETITION_STAGE_EVALUATION:
                    n_eval_seeds_actual = n_seeds  # Not applicable in competition evaluation phase, use all seeds.
                if n_eval_seeds_actual == 0:
                    skip_evaluation = True
                
                # Adjust array shapes.
                feat_scores_grid = np.zeros([n_eval_seeds_actual, len(features)], dtype=float)
                osa_scores_grid = np.zeros([n_eval_seeds_actual, 1], dtype=float)

            print("Done generating data")

        # Process termination on submission's request.
        if n_eval_seeds_actual < n_seeds and idx + 1 > n_eval_seeds_actual:
            print(
                f"\nHider evaluation terminated at {n_eval_seeds_actual} (rather than {n_seeds}) on submission request.\n"
            )
            break

        if not skip_evaluation:
            print("\n" + "=" * 80)
            print(f"Seed {idx + 1}/{n_eval_seeds_actual}: {seed}")
            print("=" * 80 + "\n")

            print("Loading generated data...")
            # Get generated data:
            hider_dir = os.path.join(args.opt_dir, "hiders", args.user)
            try:
                generated_data, generated_data_padding_mask = load_generated_data(os.path.join(hider_dir, "data.npz"))
            except (IOError, ValueError) as ex:
                raise RuntimeError(  # pylint: disable=raise-missing-from
                    "Problem loading hider generated data. "
                    f"It is possible hider solution '{args.user}' encountered an error."
                )

            print("\nCalculating feature prediction scores...")

            # Feature prediction and one-step-ahead prediction currently require imputed data.
            _, generated_data_imputed = prp.preprocess_data(generated_data, generated_data_padding_mask)
            _, test_data_imputed = prp.preprocess_data(raw_data[test_idx], raw_data_padding_mask[test_idx])

            # Feature prediction
            with tf_fixed_seed_seesion(seed):
                feat_scores, task_types = feature_prediction(
                    generated_data_imputed,
                    test_data_imputed,
                    features,
                    verbose=HIDER_EVAL_TRAINING_VERBOSE,
                    debug=HIDER_EVAL_TRAINING_DEBUG,
                )
            feat_scores_grid[idx, :] = feat_scores

            print("\nCalculating one step ahead prediction score...")

            with tf_fixed_seed_seesion(seed):
                osa_score = one_step_ahead_prediction(
                    generated_data_imputed,
                    test_data_imputed,
                    verbose=HIDER_EVAL_TRAINING_VERBOSE,
                    debug=HIDER_EVAL_TRAINING_DEBUG,
                )
            osa_scores_grid[idx, :] = osa_score

    if not skip_evaluation:

        feat_scores, osa_scores, *stds = _collapse_score_grids(feat_scores_grid, osa_scores_grid)
        print("\nHider evaluation scores:")
        _print_scores(feat_scores_grid, osa_scores_grid, feat_scores, osa_scores, *stds)

        all_scores_path = os.path.join(hider_dir, "all_scores.npz")
        np.savez(all_scores_path, feat_scores_grid=feat_scores_grid, osa_scores_grid=osa_scores_grid)

        with open(os.path.join(hider_dir, "feature_prediction_scores.txt"), "w") as f:
            for score in feat_scores:
                print(_to_str(score), file=f)
        with open(os.path.join(hider_dir, "feature_prediction_scores.task_types.txt"), "w") as f:
            for task_type in task_types:
                print(task_type, file=f)
        with open(os.path.join(hider_dir, "osa_score.txt"), "w") as f:
            print(_to_str(osa_scores), file=f)

    else:
        try:
            with open(os.path.join(hider_dir, "feature_prediction_scores.txt"), "r") as f:
                feat_scores = [float(x.strip()) for x in f.readlines()]
            with open(os.path.join(hider_dir, "feature_prediction_scores.task_types.txt"), "r") as f:
                task_types = [x.strip() for x in f.readlines()]
            with open(os.path.join(hider_dir, "osa_score.txt"), "r") as f:
                osa_score = float(f.read().strip())
            print("Using previously generated data for evaluation.")
        except OSError:
            raise RuntimeError(  # pylint: disable=raise-missing-from
                "Rescoring failed - no previous feature prediction scores / OSA prediction score found."
            )

    # Evaluate against the expected benchmark.
    eval_feat_scores, eval_task_types, eval_osa_score = _get_hider_eval_benchmarks(
        eval_dir, raw_data, raw_data_padding_mask, train_idx, test_idx, features
    )

    passed = benchmark_hider(
        feat_scores=feat_scores,
        task_types=task_types,
        osa_score=osa_score,
        eval_feat_scores=eval_feat_scores,
        eval_task_types=eval_task_types,
        eval_osa_score=eval_osa_score,
        threshold_auroc=HIDER_BENCHMARK_THRESHOLD_AUROC,
        threshold_rmse=HIDER_BENCHMARK_THRESHOLD_RMSE,
    )
    print(f'>>> Hider evaluation: {"passed" if passed else "failed"}')
    if not passed and HIDER_EVAL_RAISE_EXCEPTION:
        raise RuntimeError("Hider evaluation failed, see logs for scores.")

    # Validate hider:
    validate_hider_output(
        hider=args.user,
        hider_dir=hider_dir,
        features=features,
        data_shape=raw_data[train_idx].shape,
        raise_exception=True,
    )

    # Run hider vs seekers.
    if not COMPETITION_STAGE_EVALUATION:
        # NOTE: In competition evaluation, the vs-pairing is done on seeker runs only.
        scores = []
        seekers_list = process_vs_targets(os.path.join(args.opt_dir, "seekers"), args.vslist)
        print(f"\n>>> Will run hider '{args.user}' vs seekers: {seekers_list}\n")
        for seeker in seekers_list:
            reidentification_score = _dockerize_vs(
                client, runtime, args, seeker, args.user, is_seeker=False, seed=SEED_FOR_SEEKERS
            )
            if reidentification_score is not None:
                vs_score_file_path = os.path.join(args.opt_dir, "seekers", seeker, "vs", args.user, "vs_score.txt")
                with open(vs_score_file_path, "w") as f:
                    f.write(str(reidentification_score))
                scores.append(reidentification_score)
        if len(scores) == 0:
            score = np.float64(HIDER_SCORE_NA)
        else:
            score = np.mean(scores)
    else:
        print(
            "\nCompetition is in Evaluation stage: hider vs pairings will not be evaluated on the hider submission side.\n"
        )
        score = np.float64(HIDER_SCORE_NA)

    # Get final score.
    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print_content = (
            f"hider_score: {_to_str(score)}\n"
            f"seeker_score: {SEEKER_SCORE_NA}\n"
            f"feature_prediction_score: {_to_str(np.mean(feat_scores))}\n"
            f"one_step_ahead_score: {_to_str(osa_score)}\n"
        )
        print(print_content)
        print(print_content, file=f)


def _dockerize_seeker(client, runtime, args):
    code_dir = os.path.join(args.input_dir, "res")

    copy_dir = os.path.join(args.opt_dir, "seekers", args.user, "res")
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)
    shutil.copytree(code_dir, copy_dir)

    image = _docker_image(code_dir)
    print("Pulling {}...".format(image))
    _docker_pull(client, image)

    scores = []
    hiders_list = process_vs_targets(os.path.join(args.opt_dir, "hiders"), args.vslist)
    print(f"\n>>> Will run seeker '{args.user}' vs hiders: {hiders_list}\n")
    for hider in hiders_list:
        if os.path.exists(os.path.join(args.opt_dir, "hiders", hider, "data.npz")):
            reidentification_score = _dockerize_vs(
                client, runtime, args, args.user, hider, is_seeker=True, seed=SEED_FOR_SEEKERS
            )
            if reidentification_score is not None:
                vs_score_file_path = os.path.join(args.opt_dir, "seekers", args.user, "vs", hider, "vs_score.txt")
                with open(vs_score_file_path, "w") as f:
                    f.write(str(reidentification_score))
                scores.append(reidentification_score)
        else:
            print(f"data.npz was not found for hider '{hider}', skipping.")
    if len(scores) == 0:
        score = np.float64(SEEKER_SCORE_NA)
    else:
        score = np.mean(scores)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print_content = (
            f"hider_score: {HIDER_SCORE_NA}\n"
            f"seeker_score: {_to_str(score)}\n"
            "feature_prediction_score: 0.0\n"
            "one_step_ahead_score: 0.0\n"
        )
        print(print_content)
        print(print_content, file=f)


def _dockerize(parser, args):
    print(f"\nRunning on compute ID: {str(socket.gethostname()).split('-')[-1]}\n")

    code_dir = os.path.join(args.input_dir, "res")

    with open(os.path.join(args.input_dir, "metadata"), "r") as f:
        for line in f:
            if line.startswith("submitted-by: "):
                args.user = line[14:].strip()
                break
        else:
            parser.error("Could not determine submitting user")

    client = docker.from_env(timeout=1200)

    info = client.info()
    if "nvidia" in info["Runtimes"]:
        runtime = "nvidia"
    else:
        runtime = info["DefaultRuntime"]

    # If it doesn't exist already, create a docker network with no internet access
    try:
        client.networks.create("hide-and-seek", internal=True, check_duplicate=True)
    except docker.errors.APIError as e:
        # HTTP 409: Conflict, aka the network already existed
        if e.status_code != 409:
            raise

    is_hider = os.path.isfile(os.path.join(code_dir, "hider.py")) or os.path.isdir(os.path.join(code_dir, "hider"))
    is_seeker = os.path.isfile(os.path.join(code_dir, "seeker.py")) or os.path.isdir(os.path.join(code_dir, "seeker"))
    if is_hider and is_seeker:
        parser.error("Submission cannot be both a hider and a seeker")
    elif is_hider:
        _dockerize_hider(client, runtime, args)
    elif is_seeker:
        _dockerize_seeker(client, runtime, args)
    else:
        parser.error("Either a hider.py or seeker.py module must be present")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission.")
    parser.add_argument("--hider", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--seeker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--user", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vslist", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("-q", "--quiet", action="store_true", default=False)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("opt_dir")
    args = parser.parse_args()

    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.opt_dir = os.path.abspath(args.opt_dir)

    _dockerize(parser, args)


if __name__ == "__main__":
    main()
