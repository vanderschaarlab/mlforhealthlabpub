"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------

knn_seeker.py

Note: Find top gen_no enlarge data whose distance from 1-NN generated_data is smallest
"""

# Necessary packages
import numpy as np


def knn_seeker(generated_data, enlarged_data):
    """Find top gen_no enlarge data whose distance from 1-NN generated_data is smallest

    Args:
        - generated_data: generated data points
        - enlarged_data: train data + remaining data

    Returns:
        - reidentified_data: 1 if it is used as train data, 0 otherwise
    """

    # Parameters
    enl_no, seq_len, dim = enlarged_data.shape
    gen_no, _, _ = generated_data.shape

    # Reshape to 2d array
    enlarged_data = np.reshape(enlarged_data, [enl_no, seq_len * dim])
    generated_data = np.reshape(generated_data, [gen_no, seq_len * dim])

    # Output initialization
    distance = np.zeros([enl_no,])

    # For each data point in enlarge dataset
    for i in range(enl_no):
        temp_dist = list()
        # For each generated data point
        for j in range(gen_no):
            # Check the distance between data points in enlarge dataset and generated dataset
            tempo = np.linalg.norm(enlarged_data[i, :] - generated_data[j, :])
            temp_dist.append(tempo)
        # Find the minimum distance from 1-NN generated data
        distance[i] = np.min(temp_dist)

    # Check the threshold distance for top gen_no for 1-NN distance
    thresh = sorted(distance)[gen_no]

    # Return the decision for reidentified data
    reidentified_data = 1 * (distance <= thresh)

    return reidentified_data
