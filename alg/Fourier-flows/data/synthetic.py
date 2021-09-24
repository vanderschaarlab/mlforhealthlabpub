# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script generates sinusoidal synthetic data

"""

from __future__ import division
import numpy as np


def sine_data_generation(no, seq_len, dim, freq_scale=1):

    """Sine data generation.
  
    Args:
    
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
    Returns:
    - data: generated data
    
    """
    # Initialize the output

    data = list()

    # Generate sine data

    for i in range(no):

        # Initialize each time-series
        temp = list()

        # For each feature
        for k in range(dim):

            # Randomly drawn frequency and phase
            # freq      = np.random.uniform(0, 0.1)
            # phase     = np.random.uniform(0, 0.1)

            freq = np.random.beta(2, 2)  # np.random.uniform(0, 0.1)
            phase = np.random.normal()

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq_scale * freq * j + phase) for j in range(seq_len)]
            # temp_data = [np.exp(-1 * freq * j) * np.sin(5 * freq * j + phase) for j in range(seq_len)]
            # temp_data = [np.sinc(freq * j + phase) for j in range(seq_len)]

            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))

        # Normalize to [0,1]
        # temp = (temp + 1) * 0.5

        # Stack the generated data
        data.append(temp.reshape((-1,)))

    return data
