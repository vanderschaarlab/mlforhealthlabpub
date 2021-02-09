"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------

add_noise.py

Note: add Gaussian noise on the original data and use as the synthetic data
"""

# Necessary packages
import numpy as np


def add_noise(ori_data, noise_size):
    """Add Gaussian noise on the original data and use as the synthetic data.

    Args:
        - ori_data: original time-series data
        - noise_size: amplitude of the added noise

    Returns:
        - generated_data: generated synthetic data
    """
    # Parameters
    no, seq_len, dim = ori_data.shape

    prep_data = np.reshape(ori_data.copy(), [no * seq_len, dim])

    # Add noise
    for i in range(dim):
        noise_amplitude = np.std(prep_data[:, i]) * noise_size
        noise_vector = np.random.normal(0, noise_amplitude, size=[no * seq_len,])
        prep_data[:, i] = prep_data[:, i] + noise_vector

    generated_data = np.reshape(prep_data, [no, seq_len, dim])

    return generated_data
