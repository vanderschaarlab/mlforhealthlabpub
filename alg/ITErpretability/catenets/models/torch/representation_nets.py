import abc
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn

import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_DISC,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.torch.base import (
    DEVICE,
    BaseCATEEstimator,
    BasicNet,
    PropensityNet,
    RepresentationNet,
)
from catenets.models.torch.utils.model_utils import make_val_split

EPS = 1e-8


class BasicDragonNet(BaseCATEEstimator):
    """
    Base class for TARNet and DragonNet.

    Parameters
    ----------
    name: str
        Estimator name
    n_unit_in: int
        Number of features
    propensity_estimator: nn.Module
        Propensity estimator
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of shared & private representation layers before the hypothesis layers.
    n_units_r: int
        Number of hidden units in representation before the hypothesis layers.
    weight_decay: float
        l2 (ridge) penalty
    lr: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in the neural net. Can be 'elu', 'relu', 'selu', 'leaky_relu'.
    weighting_strategy: optional str, None
        Whether to include propensity head and which weightening strategy to use
    penalty_disc: float, default zero
         Discrepancy penalty.
    """

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        propensity_estimator: nn.Module,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        weight_decay: float = DEFAULT_PENALTY_L2,
        lr: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        nonlin: str = DEFAULT_NONLIN,
        weighting_strategy: Optional[str] = None,
        penalty_disc: float = 0,
        batch_norm: bool = True,
        early_stopping: bool = True,
        prop_loss_multiplier: float = 1,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        patience: int = DEFAULT_PATIENCE,
        dropout: bool = False,
        dropout_prob: float = 0.2
    ) -> None:
        super(BasicDragonNet, self).__init__()

        self.name = name
        self.val_split_prop = val_split_prop
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.lr = lr
        self.weight_decay = weight_decay
        self.binary_y = binary_y
        self.penalty_disc = penalty_disc
        self.early_stopping = early_stopping
        self.prop_loss_multiplier = prop_loss_multiplier
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self._repr_estimator = RepresentationNet(
            n_unit_in, n_units=n_units_r, n_layers=n_layers_r, nonlin=nonlin, batch_norm=batch_norm
        )
        self._po_estimators = []
        for idx in range(2):
            self._po_estimators.append(
                BasicNet(
                    f"{name}_po_estimator_{idx}",
                    n_units_r,
                    binary_y=binary_y,
                    n_layers_out=n_layers_out,
                    n_units_out=n_units_out,
                    nonlin=nonlin,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    dropout_prob=dropout_prob
                )
            )
        self._propensity_estimator = propensity_estimator

    def loss(
        self,
        po_pred: torch.Tensor,
        t_pred: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        discrepancy: torch.Tensor,
    ) -> torch.Tensor:
        def head_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            if self.binary_y:
                return nn.BCELoss()(y_pred, y_true)
            else:
                return (y_pred - y_true) ** 2

        def po_loss(
            po_pred: torch.Tensor, y_true: torch.Tensor, t_true: torch.Tensor
        ) -> torch.Tensor:
            y0_pred = po_pred[:, 0]
            y1_pred = po_pred[:, 1]

            loss0 = torch.mean((1.0 - t_true) * head_loss(y0_pred, y_true))
            loss1 = torch.mean(t_true * head_loss(y1_pred, y_true))

            return loss0 + loss1

        def prop_loss(t_pred: torch.Tensor, t_true: torch.Tensor) -> torch.Tensor:
            t_pred = t_pred + EPS
            return nn.CrossEntropyLoss()(t_pred, t_true)

        return (
            po_loss(po_pred, y_true, t_true) +
            self.prop_loss_multiplier*prop_loss(t_pred, t_true) + discrepancy
        )

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> "BasicDragonNet":
        """
        Fit the treatment models.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            The features to fit to
        y : torch.Tensor of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: torch.Tensor of shape (n_samples,)
            The treatment indicator
        """
        X = torch.Tensor(X).to(DEVICE)
        y = torch.Tensor(y).squeeze().to(DEVICE)
        w = torch.Tensor(w).squeeze().long().to(DEVICE)

        X, y, w, X_val, y_val, w_val, val_string = make_val_split(
            X, y, w=w, val_split_prop=self.val_split_prop, seed=self.seed
        )

        n = X.shape[0]  # could be different from before due to split

        # calculate number of batches per epoch
        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        params = (
            list(self._repr_estimator.parameters())
            + list(self._po_estimators[0].parameters())
            + list(self._po_estimators[1].parameters())
            + list(self._propensity_estimator.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # training
        val_loss_best = LARGE_VAL
        patience = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next]
                y_next = y[idx_next].squeeze()
                w_next = w[idx_next].squeeze()

                po_preds, prop_preds, discr = self._step(X_next, w_next)
                batch_loss = self.loss(po_preds, prop_preds, y_next, w_next, discr)

                batch_loss.backward()

                optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    po_preds, prop_preds, discr = self._step(X_val, w_val)
                    val_loss = self.loss(po_preds, prop_preds, y_val, w_val, discr)
                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1
                        if patience > self.patience and ((i + 1) * n_batches > self.n_iter_min):
                            break
                    if i % self.n_iter_print == 0:
                        log.info(
                        f"[{self.name}] Epoch: {i}, current {val_string} loss: {val_loss} train_loss: {torch.mean(train_loss)}"
                    )

        return self

    @abc.abstractmethod
    def _step(
        self, X: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._check_tensor(X)
        repr_preds = self._repr_estimator(X).squeeze()
        y0_preds = self._po_estimators[0](repr_preds).squeeze()
        y1_preds = self._po_estimators[1](repr_preds).squeeze()

        return torch.vstack((y0_preds, y1_preds)).T

    def predict(self, X: torch.Tensor, return_po: bool = False, training: bool = False) -> torch.Tensor:
        """
        Predict the treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        Returns
        -------
        y: array-like of shape (n_samples,)
        """
        if not training:
            self._repr_estimator.model.eval()
            self._po_estimators[0].model.eval()
            self._po_estimators[1].model.eval()

        X = self._check_tensor(X).float()
        preds = self._forward(X)
        y0_preds = preds[:, 0]
        y1_preds = preds[:, 1]

        outcome = y1_preds - y0_preds

        if return_po:
            return outcome, y0_preds, y1_preds

        return outcome

    def _maximum_mean_discrepancy(
        self, X: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        n = w.shape[0]
        n_t = torch.sum(w)

        X = X / torch.sqrt(torch.var(X, dim=0) + EPS)
        w = w.unsqueeze(dim=0)

        mean_control = (n / (n - n_t)) * torch.mean((1 - w).T * X, dim=0)
        mean_treated = (n / n_t) * torch.mean(w.T * X, dim=0)

        return self.penalty_disc * torch.sum((mean_treated - mean_control) ** 2)


class TARNet(BasicDragonNet):
    """
    Class implements Shalit et al (2017)'s TARNet
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = DEFAULT_PENALTY_DISC,
        batch_norm: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
        **kwargs: Any,
    ) -> None:
        propensity_estimator = PropensityNet(
            "tarnet_propensity_estimator",
            n_unit_in,
            2,
            "prop",
            n_layers_out_prop=n_layers_out_prop,
            n_units_out_prop=n_units_out_prop,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout_prob=dropout_prob,
            dropout=dropout
        ).to(DEVICE)
        super(TARNet, self).__init__(
            "TARNet",
            n_unit_in,
            propensity_estimator,
            binary_y=binary_y,
            nonlin=nonlin,
            penalty_disc=penalty_disc,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_prob=dropout_prob,
            **kwargs,
        )
        self.prop_loss_multiplier = 0

    def _step(
        self, X: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repr_preds = self._repr_estimator(X).squeeze()

        y0_preds = self._po_estimators[0](repr_preds).squeeze()
        y1_preds = self._po_estimators[1](repr_preds).squeeze()

        po_preds = torch.vstack((y0_preds, y1_preds)).T

        prop_preds = self._propensity_estimator(X)

        return po_preds, prop_preds, self._maximum_mean_discrepancy(repr_preds, w)


class DragonNet(BasicDragonNet):
    """
    Class implements a variant based on Shi et al (2019)'s DragonNet.
    """

    def __init__(
        self,
        n_unit_in: int,
        binary_y: bool = False,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        nonlin: str = DEFAULT_NONLIN,
        n_units_r: int = DEFAULT_UNITS_R,
        batch_norm: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
        **kwargs: Any,
    ) -> None:
        propensity_estimator = PropensityNet(
            "dragonnet_propensity_estimator",
            n_units_r,
            2,
            "prop",
            n_layers_out_prop=n_layers_out_prop,
            n_units_out_prop=n_units_out_prop,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_prob=dropout_prob
        ).to(DEVICE)
        super(DragonNet, self).__init__(
            "DragonNet",
            n_unit_in,
            propensity_estimator,
            binary_y=binary_y,
            nonlin=nonlin,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_prob=dropout_prob,
            **kwargs
        )

    def _step(
        self, X: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repr_preds = self._repr_estimator(X).squeeze()

        y0_preds = self._po_estimators[0](repr_preds).squeeze()
        y1_preds = self._po_estimators[1](repr_preds).squeeze()

        po_preds = torch.vstack((y0_preds, y1_preds)).T

        prop_preds = self._propensity_estimator(repr_preds)

        return po_preds, prop_preds, self._maximum_mean_discrepancy(repr_preds, w)
