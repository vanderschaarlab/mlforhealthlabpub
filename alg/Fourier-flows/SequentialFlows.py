# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script contains the implementation for the spectral filter module of the Fourier flow

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from filters.spectral import SpectralFilter, AttentionFilter
from fourier.transforms import DFT

from torch.distributions.multivariate_normal import MultivariateNormal

import sys

import warnings

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class FourierFlow(nn.Module):
    def __init__(self, hidden, fft_size, n_flows, FFT=True, flip=True, normalize=False):

        super().__init__()

        self.d = fft_size
        self.k = int(fft_size / 2) + 1
        self.fft_size = fft_size
        self.FFT = FFT
        self.normalize = normalize

        if flip:

            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:

            self.flips = [False for i in range(n_flows)]

        self.bijectors = nn.ModuleList(
            [SpectralFilter(self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]) for _ in range(n_flows)]
        )

        self.FourierTransform = DFT(N_fft=self.fft_size)

    def forward(self, x):

        if self.FFT:

            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std

            x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(-1, self.d + 1)

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):

        X_train = torch.from_numpy(np.array(X)).float()

        # for normalizing the spectral transforms
        X_train_spectral = self.FourierTransform(X_train)[0]
        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):

            optim.zero_grad()

            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            losses.append(loss.detach().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            if ((step % display_step) == 0) | (step == epochs - 1):

                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                print(display_string % (step, epochs, loss))

            if step == epochs - 1:

                print("Finished training!")

        return losses

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample


class RealNVP(nn.Module):
    def __init__(self, hidden, T, n_flows, flip=True, normalize=False):

        super().__init__()

        self.d = T
        self.k = int(T / 2) + 1
        self.normalize = normalize
        self.FFT = False

        if flip:

            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:

            self.flips = [False for i in range(n_flows)]

        self.bijectors = nn.ModuleList(
            [SpectralFilter(self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]) for _ in range(n_flows)]
        )

    def forward(self, x):

        if self.FFT:

            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std

            x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(-1, self.d + 1)

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):

        X_train = torch.from_numpy(np.array(X)).float()

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):

            optim.zero_grad()

            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            losses.append(loss.detach().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            if ((step % display_step) == 0) | (step == epochs - 1):

                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                print(display_string % (step, epochs, loss))

            if step == epochs - 1:

                print("Finished training!")

        return losses

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample


class TimeFlow(nn.Module):
    def __init__(self, hidden, T, n_flows, flip=True, normalize=False):

        super().__init__()

        self.d = T
        self.k = int(T / 2) + 1
        self.normalize = normalize
        self.FFT = False

        if flip:

            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:

            self.flips = [False for i in range(n_flows)]

        self.bijectors = nn.ModuleList(
            [AttentionFilter(self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]) for _ in range(n_flows)]
        )

    def forward(self, x):

        if self.FFT:

            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std

            x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(-1, self.d + 1)

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):

        X_train = torch.from_numpy(np.array(X)).float()

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):

            optim.zero_grad()

            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            losses.append(loss.detach().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            if ((step % display_step) == 0) | (step == epochs - 1):

                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                print(display_string % (step, epochs, loss))

            if step == epochs - 1:

                print("Finished training!")

        return losses

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample
