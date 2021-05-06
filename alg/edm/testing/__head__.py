import argparse
import csv
import gym
import numpy as np
import os
import sys
import pandas as pd
import pickle
import yaml

from pathlib import Path

import sys,os
sys.path.append(os.getcwd())

def get_model_path(
    env_name : str       ,
    actor    : str       ,
    run_seed : int = None,
) -> str:
    return get_path(env_name, actor, 'model.pkl', run_seed)

def get_trajs_path(
    env_name : str       ,
    actor    : str       ,
    run_seed : int = None,
) -> str:
    return get_path(env_name, actor, 'trajs.npy', run_seed)

def get_path(
    env_name : str       ,
    actor    : str       ,
    suffix   : str       ,
    run_seed : int = None,
) -> str:
    dir_path = 'volume/' + env_name
    fullpath = dir_path + '/' + actor + \
        (('_' + str(run_seed)) if run_seed is not None else '') + '_' + suffix

    Path(dir_path).mkdir(parents = True, exist_ok = True)

    return fullpath
