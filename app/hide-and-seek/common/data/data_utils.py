"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com
"""
import numpy as np
import random


def data_division(data: np.ndarray, seed: int, divide_rates: list):
    """Divide the dataset into sub datasets.

    Args:
        data (np.ndarray): Data.
        seed (int): Random seed for data division.
        divide_rates (list of float): Ratio for each division.

    Returns:
        divided_data: Divided data (list format).
        divided_index: Divided data index (list format).
    """
    # sum of the division rates should be 1
    assert sum(divide_rates) == 1

    # Output initialization
    divided_data = list()
    divided_index = list()

    # Set index
    no = len(data)
    random.seed(seed)
    np.random.seed(seed)
    index = np.random.permutation(no)

    # Set divided index & data
    for i in range(len(divide_rates)):
        temp_idx = index[int(no * sum(divide_rates[:i])) : int(no * sum(divide_rates[: (i + 1)]))]
        divided_index.append(temp_idx)

        temp_data = [data[j] for j in temp_idx]
        divided_data.append(temp_data)

    return divided_data, divided_index
