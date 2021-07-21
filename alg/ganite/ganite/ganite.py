# stdlib
from typing import Tuple

import ganite.logger as log

# third party
import numpy as np
import pandas as pd
import torch
from ganite.utils.random import enable_reproducible_results
from torch import nn

EPS = 1e-10


class CounterfactualGenerator(nn.Module):
    """
    The counterfactual generator, G, uses the feature vector x,
    the treatment vector t, and the factual outcome yf, to generate
    a potential outcome vector, hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int
    ) -> None:
        super(CounterfactualGenerator, self).__init__()
        # Generator Layer
        hidden = [
            nn.Linear(DimHidden, DimHidden),
            nn.LeakyReLU(),
        ] * depth
        self.common = nn.Sequential(
            nn.Linear(
                Dim + 2, DimHidden
            ),  # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector      (Z)
            nn.LeakyReLU(),
            *hidden,
        )

        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, 1),
                    nn.Sigmoid(),
                )
            )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([x, t, y], dim=1)
        G_h2 = self.common(inputs)

        G_probs = []
        for out in self.outs:
            res = out(G_h2) + EPS
            G_probs.append(res)

        G_prob = torch.cat(G_probs, dim=-1)

        return G_prob


class CounterfactualDiscriminator(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(self, Dim: int, Treatments: list, DimHidden: int, depth: int) -> None:
        super(CounterfactualDiscriminator, self).__init__()
        hidden = [
            nn.Linear(DimHidden, DimHidden),
            nn.LeakyReLU(),
        ] * depth

        self.Treatments = Treatments

        self.model = nn.Sequential(
            nn.Linear(Dim + len(Treatments), DimHidden),
            nn.LeakyReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, hat_y: torch.Tensor
    ) -> torch.Tensor:
        # Factual & Counterfactual outcomes concatenate

        raw = [x]
        for idx, tidx in enumerate(self.Treatments):
            raw.append(
                (t == tidx).int() * y
                + (t != tidx).int() * hat_y[:, idx].reshape([-1, 1])
            )
        inputs = torch.cat(raw, dim=-1)

        return self.model(inputs)


class InferenceNets(nn.Module):
    """
    The ITE generator uses only the feature vector, x, to generate a potential outcome vector hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int
    ) -> None:
        super(InferenceNets, self).__init__()
        hidden = [
            nn.Linear(DimHidden, DimHidden),
            nn.LeakyReLU(),
        ] * depth
        self.common = nn.Sequential(
            nn.Linear(Dim, DimHidden),
            nn.LeakyReLU(),
            *hidden,
        )
        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, 1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        I_h = self.common(x)

        I_probs = []
        for out in self.outs:
            I_probs.append(out(I_h))

        return torch.sigmoid(torch.cat(I_probs, dim=1))


class Ganite:
    """
    The GANITE framework generates potential outcomes for a given feature vector x.
    It consists of 2 components:
     - The Counterfactual Generator block(generator + discriminator)
     - The ITE block(InferenceNets).
    """

    def __init__(
        self,
        X: np.ndarray,
        Treatments: np.ndarray,
        Y: np.ndarray,
        dim_hidden: int = 40,
        alpha: float = 1,
        beta: float = 1,
        minibatch_size: int = 256,
        depth: int = 8,
        num_iterations: int = 5000,
        num_discr_iterations: int = 10,
    ) -> None:
        X = np.asarray(X)
        Treatments = np.asarray(Treatments)
        Y = np.asarray(Y)

        if np.isnan(np.sum(X)):
            raise ValueError("X contains NaNs")
        if len(X) != len(Treatments):
            raise ValueError("Features/Treatments mismatch")
        if len(X) != len(Y):
            raise ValueError("Features/Labels mismatch")

        enable_reproducible_results()

        dim_in = X.shape[1]
        self.original_treatments = np.sort(np.unique(Treatments))
        self.treatments = [0, 1]

        if len(self.original_treatments) != 2:
            raise ValueError("Only two treatment categories supported")

        # Hyperparameters
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.num_iterations = num_iterations
        self.num_discr_iterations = num_discr_iterations

        # Layers
        self.counterfactual_generator = CounterfactualGenerator(
            dim_in, len(self.treatments), dim_hidden, depth
        )
        self.counterfactual_discriminator = CounterfactualDiscriminator(
            dim_in, self.treatments, dim_hidden, depth
        )
        self.inference_nets = InferenceNets(
            dim_in, len(self.treatments), dim_hidden, depth
        )

        # Solvers
        self.DG_solver = torch.optim.Adam(
            list(self.counterfactual_generator.parameters())
            + list(self.counterfactual_discriminator.parameters()),
            lr=1e-4,
            eps=10e-8,
            weight_decay=1e-3,
        )
        self.I_solver = torch.optim.Adam(
            self.inference_nets.parameters(), lr=1e-4, weight_decay=1e-3
        )

        self._fit(X, Treatments, Y)

    def _sample_minibatch(
        self, X: torch.Tensor, T: torch.tensor, Y: torch.Tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        idx_mb = np.random.randint(0, X.shape[0], self.minibatch_size)

        X_mb = X[idx_mb, :]
        T_mb = T[idx_mb].reshape([self.minibatch_size, 1])
        Y_mb = Y[idx_mb].reshape([self.minibatch_size, 1])

        return X_mb, T_mb, Y_mb

    def _fit(
        self,
        X: pd.DataFrame,
        Treatment: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> "Ganite":
        Train_X = torch.from_numpy(X).float()
        Train_T = torch.from_numpy(Treatment).float().reshape([-1, 1])
        Train_Y = torch.from_numpy(Y).float().reshape([-1, 1])
        # Encode
        min_t_val = Train_T.min()
        Train_T = (Train_T > min_t_val).float()

        # Iterations
        # Train G and D first
        self.counterfactual_generator.train()
        self.counterfactual_discriminator.train()
        self.inference_nets.train()

        for it in range(self.num_iterations):
            self.DG_solver.zero_grad()

            X_mb, T_mb, Y_mb = self._sample_minibatch(Train_X, Train_T, Train_Y)

            for kk in range(self.num_discr_iterations):
                self.DG_solver.zero_grad()

                Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb).clone()
                D_out = self.counterfactual_discriminator(X_mb, T_mb, Y_mb, Tilde)

                if torch.isnan(Tilde).any():
                    raise RuntimeError("counterfactual_generator generated NaNs")
                if torch.isnan(D_out).any():
                    raise RuntimeError("counterfactual_discriminator generated NaNs")

                D_loss = nn.BCELoss()(D_out, T_mb)

                D_loss.backward()
                self.DG_solver.step()

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)
            D_out = self.counterfactual_discriminator(X_mb, T_mb, Y_mb, Tilde)
            D_loss = nn.BCELoss()(D_out, T_mb)

            G_loss_GAN = D_loss

            gen_loss = torch.zeros(Y_mb.shape)

            for idx, tidx in enumerate(self.treatments):
                gen_loss += (T_mb == tidx).int() * Tilde[:, idx].reshape([-1, 1])

            G_loss_R = torch.mean(nn.MSELoss()(Y_mb, gen_loss))

            G_loss = G_loss_R + self.alpha * G_loss_GAN

            if it % 100 == 0:
                log.debug(f"Generator loss epoch {it}: {D_loss} {G_loss}")
                if torch.isnan(D_loss).any():
                    raise RuntimeError("counterfactual_discriminator generated NaNs")

                if torch.isnan(G_loss).any():
                    raise RuntimeError("counterfactual_generator generated NaNs")

            G_loss.backward()
            self.DG_solver.step()

        # Train I and ID
        for it in range(self.num_iterations):
            self.I_solver.zero_grad()

            X_mb, T_mb, Y_mb = self._sample_minibatch(Train_X, Train_T, Train_Y)

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)

            hat = self.inference_nets(X_mb)

            I_loss1: torch.Tensor = 0
            I_loss2: torch.Tensor = 0

            for idx, tidx in enumerate(self.treatments):
                I_loss1 += torch.mean(
                    nn.MSELoss()(
                        (T_mb == tidx).int() * Y_mb
                        + (T_mb != tidx).int() * Tilde[:, idx].reshape([-1, 1]),
                        hat[:, idx].reshape([-1, 1]),
                    )
                )
            for idx, tidx in enumerate(self.treatments):
                I_loss2 += torch.mean(
                    nn.MSELoss()(
                        (T_mb != tidx).int() * Y_mb
                        + (T_mb == tidx).int() * Tilde[:, idx].reshape([-1, 1]),
                        hat[:, idx].reshape([-1, 1]),
                    )
                )
            I_loss = I_loss1 + self.beta * I_loss2

            if it % 100 == 0:
                log.debug(f"Inference loss epoch {it}: {I_loss}")

            I_loss.backward()
            self.I_solver.step()

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        with torch.no_grad():
            X = torch.from_numpy(np.asarray(X)).float()
            y_hat = self.inference_nets(X).detach().numpy()

        return pd.DataFrame(y_hat[:, 1] - y_hat[:, 0])
