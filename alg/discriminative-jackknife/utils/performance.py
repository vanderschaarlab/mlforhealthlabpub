

# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------

from matplotlib import pyplot as plt

def plot_1D_uncertainty(results, Y_test, data_index):
    
    plt.fill_between(list(range(len(results["Lower limit"][data_index]))), 
                     results["Lower limit"][data_index].reshape(-1,), 
                     results["Upper limit"][data_index].reshape(-1,), color="r", alpha=0.25)

    plt.plot(results["Lower limit"][data_index], linestyle=":", linewidth=3, color="r")
    plt.plot(results["Upper limit"][data_index], linestyle=":", linewidth=3, color="r")

    plt.plot(Y_test[data_index], linestyle="--", linewidth=2, color="black")
    plt.plot(results["Point predictions"][data_index], linewidth=3, color="r", Marker="o")
