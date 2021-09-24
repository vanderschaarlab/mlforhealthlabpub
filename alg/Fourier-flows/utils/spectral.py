# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from __future__ import absolute_import, division, print_function
import numpy as np


def get_frequencies(xsampled):

    freqs = [
        (50 - np.argmax(np.abs(np.fft.fftshift(np.fft.fft(xsampled[k][:51]))))) / 100 for k in range(len(xsampled))
    ]

    return freqs


def power_spectral_density(x_time):

    x_time = [(x_time[k] - np.mean(x_time[k])) / np.max(x_time[k] - np.mean(x_time[k])) for k in range(len(x_time))]

    Xcorr = [np.correlate(x_time[k], x_time[k], "full") for k in range(len(x_time))]
    Rcorr = np.mean(np.array(Xcorr), axis=0)
    PSD = np.abs(np.fft.fftshift(np.fft.fft(Rcorr))) / len(Rcorr)

    return PSD
