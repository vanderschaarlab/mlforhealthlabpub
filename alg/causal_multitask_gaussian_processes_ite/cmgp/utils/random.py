# stdlib
import random

# third party
import numpy as np
import torch


def enable_reproducible_results() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
