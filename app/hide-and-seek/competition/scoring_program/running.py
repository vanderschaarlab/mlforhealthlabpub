#!/usr/bin/env python3

import argparse
import os
import sys
import logging

import numpy as np

import computils.debug as dbg
from computils.misc import fix_all_random_seeds, read_competition_config, dump_redacted_exc, time_limited
from computils.solutions import load_data, load_generated_data, parse_hider_output

competition_config = read_competition_config("/opt/hide-and-seek/competition_config.json")
MAX_SEQ_LEN = competition_config.get("MAX_SEQ_LEN", 100)
TRAIN_RATE = competition_config.get("TRAIN_RATE", 0.5)
DATA_FILE_NAME = competition_config.get("DATA_FILE_NAME", "test_longitudinal_data.csv")
FORCE_REPROCESS = competition_config.get("FORCE_REPROCESS", False)

SEED = competition_config.get("SEED", 12345)
fix_all_random_seeds(SEED)

SOLUTIONS_VERBOSE_DEBUG = competition_config.get("SOLUTIONS_VERBOSE_DEBUG", False)
SOLUTION_TIME_LIMIT = competition_config.get("SOLUTION_TIME_LIMIT", 86400)
LOG_LEVEL = logging.DEBUG if SOLUTIONS_VERBOSE_DEBUG else logging.INFO
logger = dbg.setup_logger(name="scoring_logger", level=LOG_LEVEL, format_str="%(message)s")
dbg.set_log_method(logger.debug)
DEBUG_DATA = competition_config.get("DEBUG_DATA", False)


def _run_hider(args):

    hider_dir = os.path.join(args.opt_dir, "hiders", args.user)
    exc_dump = os.path.join(hider_dir, "EXCEPTION")

    msg_to_scorer = os.path.join(hider_dir, "MSG")
    if os.path.exists(msg_to_scorer):
        os.remove(msg_to_scorer)

    with dump_redacted_exc(exc_dump):

        print("Loading data...")
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

        print("Running hider...")

        train_data = raw_data[train_idx]
        padding_mask = raw_data_padding_mask[train_idx]

        import hider as hider_module  # pylint: disable=import-error

        hider_input = {"data": train_data, "seed": args.seed, "padding_mask": padding_mask}

        # Execute hider
        hider_output = hider_module.hider(hider_input)

        if hider_output in ("reevaluate", "rescore"):
            with open(msg_to_scorer, "w") as f:
                f.write(hider_output)
            return

        generated_data, generated_data_padding_mask, eval_seeds = parse_hider_output(hider_output)

        if eval_seeds is not None:
            with open(msg_to_scorer, "w") as f:
                f.write(str(eval_seeds))

        dbg.ar(train_data, "train_data")
        dbg.ar(generated_data, "generated_data")

        print("Hider done")

        np.savez(
            os.path.join(hider_dir, "data.npz"),
            generated_data=generated_data,
            padding_mask=generated_data_padding_mask if generated_data_padding_mask is not None else [],
        )
        print("Saved hider output")


def _run_seeker(args):

    vs_dir = os.path.join(args.opt_dir, "seekers", args.user, "vs", args.vs)
    exc_dump = os.path.join(vs_dir, "EXCEPTION")

    with dump_redacted_exc(exc_dump):

        print("Loading data...")
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

        logger.debug(f"raw_data shape: {raw_data.shape}")

        data_path = os.path.join(args.opt_dir, "hiders", args.vs, "data.npz")
        generated_data, generated_data_padding_mask = load_generated_data(data_path)

        logger.debug(f"generated_data shape: {generated_data.shape}")

        print("Running seeker...")

        import seeker as seeker_module  # pylint: disable=import-error

        seeker_input = {
            "generated_data": generated_data,
            "enlarged_data": raw_data,
            "seed": args.seed,
            "generated_data_padding_mask": generated_data_padding_mask,
            "enlarged_data_padding_mask": raw_data_padding_mask,
        }

        # Execute seeker
        reidentified_data = seeker_module.seeker(seeker_input)

        print("Seeker done")

        dbg.ar(reidentified_data, "reidentified_data")

        np.savez(
            os.path.join(vs_dir, "data.npz"), reidentified_data=reidentified_data,
        )
        print("Saved seeker output")


def _run(parser, args):
    code_dir = os.path.join(args.input_dir, "res")
    os.chdir(code_dir)
    sys.path = [code_dir] + sys.path

    if args.hider and args.seeker:
        parser.error("Can't be both a hider and a seeker")
    elif args.hider:
        exc_dump = os.path.join(args.opt_dir, "hiders", args.user, "EXCEPTION")
        time_limited(_run_hider, SOLUTION_TIME_LIMIT, exc_dump, args)
    elif args.seeker:
        exc_dump = os.path.join(args.opt_dir, "seekers", args.user, "vs", args.vs, "EXCEPTION")
        time_limited(_run_seeker, SOLUTION_TIME_LIMIT, exc_dump, args)
    else:
        parser.error("Must be either a hider or a seeker")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission.")
    parser.add_argument("--hider", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--seeker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--user", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vs", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", required=True, type=int, help=argparse.SUPPRESS)
    parser.add_argument("input_dir")
    parser.add_argument("opt_dir")
    args = parser.parse_args()

    args.input_dir = os.path.abspath(args.input_dir)
    args.opt_dir = os.path.abspath(args.opt_dir)

    _run(parser, args)


if __name__ == "__main__":
    main()
