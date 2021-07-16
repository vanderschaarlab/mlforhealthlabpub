"""
Author: Alicia Curth
File to run experiments from shell
"""
import argparse
import sys
from typing import Any

import catenets.logger as log
from catenets.experiments.ihdp_experiments import do_ihdp_experiments
from catenets.experiments.simulations_AISTATS import main_AISTATS

log.add(sink=sys.stderr, level="DEBUG")


def init_arg() -> Any:
    # arg parser if script is run from shell
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="ihdp", type=str)
    parser.add_argument("--setting", default=1, type=int)
    parser.add_argument("--models", default=None, type=str)
    parser.add_argument("--file_name", default="results", type=str)
    parser.add_argument("--n_repeats", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()
    if args.experiment == "simulation":
        main_AISTATS(
            setting=args.setting,
            models=args.models,
            file_name=args.file_name,
            n_repeats=args.n_repeats,
        )
    elif args.experiment == "ihdp":
        do_ihdp_experiments(
            models=args.models, file_name=args.file_name, n_exp=args.n_repeats
        )
