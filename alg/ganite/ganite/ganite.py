# stdlib
from typing import Tuple

import ganite.logger as log

# third party
import numpy as np
import torch
from ganite.utils.random import enable_reproducible_results
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10


class CounterfactualGenerator(nn.Module):
    """
    The counterfactual generator, G, uses the feature vector x,
    the treatment vector t, and the factual outcome yf, to generate
    a potential outcome vector, hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int, binary_y: bool
    ) -> None:
        super(CounterfactualGenerator, self).__init__()
        # Generator Layer
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Dropout(),
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.common = nn.Sequential(
            nn.Linear(
                Dim + 2, DimHidden
            ),  # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector      (Z)
            nn.LeakyReLU(),
            *hidden,
        ).to(DEVICE)

        self.binary_y = binary_y
        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, 1),
                ).to(DEVICE)
            )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([x, t, y], dim=1).to(DEVICE)

        G_h2 = self.common(inputs)

        G_prob1 = self.outs[0](G_h2)
        G_prob2 = self.outs[1](G_h2)

        G_prob = torch.cat([G_prob1, G_prob2], dim=1).to(DEVICE)

        if self.binary_y:
            return torch.sigmoid(G_prob)
        else:
            return G_prob


class CounterfactualDiscriminator(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(
        self, Dim: int, Treatments: list, DimHidden: int, depth: int, binary_y: bool
    ) -> None:
        super(CounterfactualDiscriminator, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.Treatments = Treatments

        self.model = nn.Sequential(
            nn.Linear(Dim + len(Treatments), DimHidden),
            nn.LeakyReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, hat_y: torch.Tensor
    ) -> torch.Tensor:
        # Factual & Counterfactual outcomes concatenate
        inp0 = (1.0 - t) * y + t * hat_y[:, 0].reshape([-1, 1])
        inp1 = t * y + (1.0 - t) * hat_y[:, 1].reshape([-1, 1])

        inputs = torch.cat([x, inp0, inp1], dim=1).to(DEVICE)
        return self.model(inputs)


class InferenceNets(nn.Module):
    """
    The ITE generator uses only the feature vector, x, to generate a potential outcome vector hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int, binary_y: bool
    ) -> None:
        super(InferenceNets, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.common = nn.Sequential(
            nn.Linear(Dim, DimHidden),
            nn.LeakyReLU(),
            *hidden,
        ).to(DEVICE)
        self.binary_y = binary_y

        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, 1),
                ).to(DEVICE)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        I_h = self.common(x)

        I_probs = []
        for out in self.outs:
            I_probs.append(out(I_h))

        if self.binary_y:
            return torch.sigmoid(torch.cat(I_probs, dim=1).to(DEVICE))
        else:
            return torch.cat(I_probs, dim=1).to(DEVICE)


class Ganite(nn.Module):
    """
    The GANITE framework generates potential outcomes for a given feature vector x.
    It consists of 2 components:
     - The Counterfactual Generator block(generator + discriminator)
     - The ITE block(InferenceNets).
    """

    def __init__(
        self,
        X: torch.Tensor,
        Treatments: torch.Tensor,
        Y: torch.Tensor,
        dim_hidden: int = 100,
        alpha: float = 0.1,
        beta: float = 0,
        minibatch_size: int = 256,
        depth: int = 0,
        num_iterations: int = 5000,
        num_discr_iterations: int = 1,
    ) -> None:
        super(Ganite, self).__init__()

        X = self._check_tensor(X)
        Treatments = self._check_tensor(Treatments)
        Y = self._check_tensor(Y)

        if np.isnan(np.sum(X.cpu().numpy())):
            raise ValueError("X contains NaNs")
        if len(X) != len(Treatments):
            raise ValueError("Features/Treatments mismatch")
        if len(X) != len(Y):
            raise ValueError("Features/Labels mismatch")

        enable_reproducible_results()

        dim_in = X.shape[1]
        self.original_treatments = np.sort(np.unique(Treatments.cpu().numpy()))
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

        binary_y = len(np.unique(Y.cpu().numpy())) == 2
        # Layers
        self.counterfactual_generator = CounterfactualGenerator(
            dim_in, len(self.treatments), dim_hidden, depth, binary_y
        ).to(DEVICE)
        self.counterfactual_discriminator = CounterfactualDiscriminator(
            dim_in, self.treatments, dim_hidden, depth, binary_y
        ).to(DEVICE)
        self.inference_nets = InferenceNets(
            dim_in, len(self.treatments), dim_hidden, depth, binary_y
        ).to(DEVICE)

        # Solvers
        self.DG_solver = torch.optim.Adam(
            list(self.counterfactual_generator.parameters())
            + list(self.counterfactual_discriminator.parameters()),
            lr=1e-3,
            eps=1e-8,
            weight_decay=1e-3,
        )
        self.I_solver = torch.optim.Adam(
            self.inference_nets.parameters(), lr=1e-3, weight_decay=1e-3
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
        X: torch.Tensor,
        Treatment: torch.Tensor,
        Y: torch.Tensor,
    ) -> "Ganite":
        Train_X = self._check_tensor(X).float()
        Train_T = self._check_tensor(Treatment).float().reshape([-1, 1])
        Train_Y = self._check_tensor(Y).float().reshape([-1, 1])
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

            G_loss_R = torch.mean(
                nn.MSELoss()(
                    Y_mb,
                    T_mb * Tilde[:, 1].reshape([-1, 1])
                    + (1.0 - T_mb) * Tilde[:, 0].reshape([-1, 1]),
                )
            )

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

            I_loss1 = torch.mean(
                nn.MSELoss()(
                    T_mb * Y_mb + (1 - T_mb) * Tilde[:, 1].reshape([-1, 1]),
                    hat[:, 1].reshape([-1, 1]),
                )
            )
            I_loss2 = torch.mean(
                nn.MSELoss()(
                    (1 - T_mb) * Y_mb + T_mb * Tilde[:, 0].reshape([-1, 1]),
                    hat[:, 0].reshape([-1, 1]),
                )
            )
            I_loss = I_loss1 + self.beta * I_loss2

            if it % 100 == 0:
                log.debug(f"Inference loss epoch {it}: {I_loss}")

            I_loss.backward()
            self.I_solver.step()

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            X = self._check_tensor(X).float()
            y_hat = self.inference_nets(X).detach()

        return y_hat[:, 1] - y_hat[:, 0]

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)
