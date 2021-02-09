"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 21th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------

binary_predictor.py

Note: Make binary predictor that predict synthetic data from original enlarged data.
      Then, use the predicted scores as the distance between synthetic and real data
"""

# Necessary packages
import numpy as np

# Resolve general_rnn module.
try:
    from computils.general_rnn import GeneralRNN  # pylint: disable=import-error,no-name-in-module
except ModuleNotFoundError:
    try:
        from utils.general_rnn import GeneralRNN  # type: ignore
    except ModuleNotFoundError:
        from .general_rnn import GeneralRNN  # type: ignore  # pylint: disable=relative-beyond-top-level


def binary_predictor(generated_data, enlarged_data, verbose=False):
    """Find top gen_no enlarge data whose predicted scores is largest using the trained predictor.

    Args:
        - generated_data: generated data points
        - enlarged_data: train data + remaining data

    Returns:
        - reidentified_data: 1 if it is used as train data, 0 otherwise
    """

    # Parameters
    enl_no, seq_len, dim = enlarged_data.shape
    gen_no, _, _ = generated_data.shape

    # Set model parameters
    model_parameters = {
        "task": "classification",
        "model_type": "gru",
        "h_dim": dim,
        "n_layer": 3,
        "batch_size": 128,
        "epoch": 20,
        "learning_rate": 0.001,
    }

    # Set training features and labels
    train_x = np.concatenate((generated_data.copy(), enlarged_data.copy()), axis=0)
    train_y = np.concatenate((np.zeros([gen_no, 1]), np.ones([enl_no, 1])), axis=0)

    idx = np.random.permutation(enl_no + gen_no)
    train_x = train_x[idx, :, :]
    train_y = train_y[idx, :]

    # Train the binary predictor
    general_rnn = GeneralRNN(model_parameters)
    general_rnn.fit(train_x, train_y, verbose=verbose)

    # Measure the distance from synthetic data using the trained model
    distance = general_rnn.predict(enlarged_data)

    # Check the threshold distance for top gen_no for 1-NN distance
    thresh = sorted(distance)[gen_no]

    # Return the decision for reidentified data
    reidentified_data = 1 * (distance <= thresh)

    return reidentified_data
