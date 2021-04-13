"""
Author: Alicia Curth
Implements Pseudo-outcome based Two-step Nets, namely the DR-learner, the PW-learner and the
RA-learner.
"""
import numpy as onp
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from catenets.models.base import BaseCATENet, train_output_net_only
from catenets.models.model_utils import check_shape_1d_data, check_X_is_np
from catenets.models.transformation_utils import _get_transformation_function, \
    DR_TRANSFORMATION, PW_TRANSFORMATION, RA_TRANSFORMATION
from catenets.models.representation_nets import train_snet1, predict_snet1, train_snet2, \
    predict_snet2
from catenets.models.disentangled_nets import train_snet3, predict_snet3, \
    DEFAULT_UNITS_R_SMALL_S3, DEFAULT_UNITS_R_BIG_S3
from catenets.models.snet import train_snet, predict_snet, DEFAULT_UNITS_R_BIG_S, \
    DEFAULT_UNITS_R_SMALL_S
from catenets.models.constants import *

T_STRATEGY = 'T'
S1_STRATEGY = 'S1'
S2_STRATEGY = 'S2'
S3_STRATEGY = 'S3'
S_STRATEGY = 'S'

ALL_STRATEGIES = [T_STRATEGY, S1_STRATEGY, S2_STRATEGY, S3_STRATEGY, S_STRATEGY]


class PseudoOutcomeNet(BaseCATENet):
    """
    Class implements TwoStepLearners based on pseudo-outcome regression as discussed in
    Curth &vd Schaar (2021): RA-learner, PW-learner and DR-learner

    Parameters
    ----------
    first_stage_strategy: str, default 't'
        which nuisance estimator to use in first stage
    data_split: bool, default False
        Whether to split the data in two folds for estimation
    cross_fit: bool, default False
        Whether to perform cross fitting
    n_cf_folds: int
        Number of crossfitting folds to use
    transformation: str, default 'AIPW'
        pseudo-outcome to use ('AIPW' for DR-learner, 'HT' for PW learner, 'RA' for RA-learner)
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        First stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        First stage Number of hidden units in each hypothesis layer
    n_layers_r: int
        First stage Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r: int
        First stage Number of hidden units in each representation layer
    n_layers_out_t: int
        Second stage Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out_t: int
        Second stage Number of hidden units in each hypothesis layer
    n_layers_r_t: int
        Second stage Number of representation layers before hypothesis layers (distinction between
        hypothesis layers and representation layers is made to match TARNet & SNets)
    n_units_r_t: int
        Second stage Number of hidden units in each representation layer
    penalty_l2: float
        First stage l2 (ridge) penalty
    penalty_l2_t: float
        Second stage l2 (ridge) penalty
    step_size: float
        First stage learning rate for optimizer
    step_size_t: float
        Second stage learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    early_stopping: bool, default True
        Whether to use early stopping
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    verbose: int, default 1
        Whether to print notifications
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    """
    def __init__(self, first_stage_strategy: str = T_STRATEGY,
                 data_split: bool = False,
                 cross_fit: bool = False, n_cf_folds: int = DEFAULT_CF_FOLDS,
                 transformation: str = DR_TRANSFORMATION,
                 binary_y: bool = False,
                 n_layers_out: int = DEFAULT_LAYERS_OUT,
                 n_layers_r: int = DEFAULT_LAYERS_R,
                 n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
                 n_layers_r_t: int = DEFAULT_LAYERS_R_T,
                 n_units_out: int = DEFAULT_UNITS_OUT,
                 n_units_r: int = DEFAULT_UNITS_R,
                 n_units_out_t: int = DEFAULT_UNITS_OUT_T,
                 n_units_r_t: int = DEFAULT_UNITS_R_T,
                 penalty_l2: float = DEFAULT_PENALTY_L2,
                 penalty_l2_t: float = DEFAULT_PENALTY_L2,
                 step_size: float = DEFAULT_STEP_SIZE,
                 step_size_t: float = DEFAULT_STEP_SIZE_T,
                 n_iter: int = DEFAULT_N_ITER,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 n_iter_min: int = DEFAULT_N_ITER_MIN,
                 val_split_prop: float = DEFAULT_VAL_SPLIT,
                 early_stopping: bool = True,
                 patience: int = DEFAULT_PATIENCE,
                 verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                 seed: int = DEFAULT_SEED,
                 rescale_transformation: bool = False,
                 penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                 n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                 nonlin: str = DEFAULT_NONLIN
                 ):
        # settings
        self.first_stage_strategy = first_stage_strategy
        self.binary_y = binary_y
        self.transformation = transformation
        self.data_split = data_split
        self.cross_fit = cross_fit
        self.n_cf_folds = n_cf_folds

        # model architecture hyperparams
        self.n_layers_out = n_layers_out
        self.n_layers_out_t = n_layers_out_t
        self.n_layers_r = n_layers_r
        self.n_layers_r_t = n_layers_r_t
        self.n_units_r_small = n_units_r_small
        self.n_units_out = n_units_out
        self.n_units_out_t = n_units_out_t
        self.n_units_r = n_units_r
        self.n_units_r_t = n_units_r_t
        self.nonlin = nonlin

        # other hyperparameters
        self.penalty_l2 = penalty_l2
        self.penalty_l2_t = penalty_l2_t
        self.penalty_orthogonal = penalty_orthogonal
        self.step_size = step_size
        self.step_size_t = step_size_t
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.rescale_transformation = rescale_transformation

    def _get_train_function(self):
        return train_pseudooutcome_net

    def fit(self, X, y, w, p=None):
        # overwrite super so we can pass p as extra param
        # some quick input checks
        X = check_X_is_np(X)
        self._check_inputs(w, p)

        train_func = self._get_train_function()
        train_params = self.get_params()

        if 'transformation' not in train_params.keys():
            train_params.update({'transformation': self.transformation})

        if self.rescale_transformation:
            self._params, self._predict_funs, self._scale_factor = train_func(X, y, w, p,
                                                                              **train_params)
        else:
            self._params, self._predict_funs = train_func(X, y, w, p, **train_params)

    def _get_predict_function(self):
        # Two step nets do not need this
        pass

    def predict(self, X, return_po: bool = False, return_prop: bool = False):
        # check input
        if return_po:
            raise NotImplementedError('TwoStepNets have no Potential outcome predictors.')

        if return_prop:
            raise NotImplementedError('TwoStepNets have no Propensity predictors.')

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.rescale_transformation:
            return 1 / self._scale_factor * self._predict_funs(self._params, X)
        else:
            return self._predict_funs(self._params, X)


class DRNet(PseudoOutcomeNet):
    """ Wrapper for DR-learner using PseudoOutcomeNet"""
    def __init__(self, first_stage_strategy: str = T_STRATEGY,
                 data_split: bool = False,
                 cross_fit: bool = False, n_cf_folds: int = DEFAULT_CF_FOLDS,
                 binary_y: bool = False,
                 n_layers_out: int = DEFAULT_LAYERS_OUT,
                 n_layers_r: int = DEFAULT_LAYERS_R,
                 n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
                 n_layers_r_t: int = DEFAULT_LAYERS_R_T,
                 n_units_out: int = DEFAULT_UNITS_OUT,
                 n_units_r: int = DEFAULT_UNITS_R,
                 n_units_out_t: int = DEFAULT_UNITS_OUT_T,
                 n_units_r_t: int = DEFAULT_UNITS_R_T,
                 penalty_l2: float = DEFAULT_PENALTY_L2,
                 penalty_l2_t: float = DEFAULT_PENALTY_L2,
                 step_size: float = DEFAULT_STEP_SIZE,
                 step_size_t: float = DEFAULT_STEP_SIZE_T,
                 n_iter: int = DEFAULT_N_ITER,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 n_iter_min: int = DEFAULT_N_ITER_MIN,
                 val_split_prop: float = DEFAULT_VAL_SPLIT,
                 early_stopping: bool = True,
                 patience: int = DEFAULT_PATIENCE,
                 verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                 seed: int = DEFAULT_SEED,
                 rescale_transformation: bool = False,
                 penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                 n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                 nonlin: str = DEFAULT_NONLIN
                 ):
        super().__init__(first_stage_strategy=first_stage_strategy, data_split=data_split,
                         cross_fit=cross_fit, n_cf_folds=n_cf_folds,
                         transformation=DR_TRANSFORMATION, binary_y=binary_y,
                         n_layers_out=n_layers_out, n_layers_r=n_layers_r,
                         n_layers_out_t=n_layers_out_t, n_layers_r_t=n_layers_r_t,
                         n_units_out=n_units_out,n_units_r=n_units_r, n_units_out_t=n_units_out_t,
                         n_units_r_t=n_units_r_t, penalty_l2=penalty_l2, penalty_l2_t=penalty_l2_t,
                         step_size=step_size, step_size_t=step_size_t, n_iter=n_iter,
                         batch_size=batch_size, n_iter_min=n_iter_min,
                         val_split_prop=val_split_prop, early_stopping=early_stopping,
                         patience=patience, verbose=verbose, n_iter_print=n_iter_print,
                         seed=seed, penalty_orthogonal=penalty_orthogonal,
                         n_units_r_small=n_units_r_small, nonlin=nonlin,
                         rescale_transformation=rescale_transformation)


class RANet(PseudoOutcomeNet):
    """ Wrapper for RA-learner using PseudoOutcomeNet"""
    def __init__(self, first_stage_strategy: str = T_STRATEGY,
                 data_split: bool = False,
                 cross_fit: bool = False, n_cf_folds: int = DEFAULT_CF_FOLDS,
                 binary_y: bool = False,
                 n_layers_out: int = DEFAULT_LAYERS_OUT,
                 n_layers_r: int = DEFAULT_LAYERS_R,
                 n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
                 n_layers_r_t: int = DEFAULT_LAYERS_R_T,
                 n_units_out: int = DEFAULT_UNITS_OUT,
                 n_units_r: int = DEFAULT_UNITS_R,
                 n_units_out_t: int = DEFAULT_UNITS_OUT_T,
                 n_units_r_t: int = DEFAULT_UNITS_R_T,
                 penalty_l2: float = DEFAULT_PENALTY_L2,
                 penalty_l2_t: float = DEFAULT_PENALTY_L2,
                 step_size: float = DEFAULT_STEP_SIZE,
                 step_size_t: float = DEFAULT_STEP_SIZE_T,
                 n_iter: int = DEFAULT_N_ITER,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 n_iter_min: int = DEFAULT_N_ITER_MIN,
                 val_split_prop: float = DEFAULT_VAL_SPLIT,
                 early_stopping: bool = True,
                 patience: int = DEFAULT_PATIENCE,
                 verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                 seed: int = DEFAULT_SEED,
                 rescale_transformation: bool = False,
                 penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                 n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                 nonlin: str = DEFAULT_NONLIN
                 ):
        super().__init__(first_stage_strategy=first_stage_strategy, data_split=data_split,
                         cross_fit=cross_fit, n_cf_folds=n_cf_folds,
                         transformation=RA_TRANSFORMATION, binary_y=binary_y,
                         n_layers_out=n_layers_out, n_layers_r=n_layers_r,
                         n_layers_out_t=n_layers_out_t, n_layers_r_t=n_layers_r_t,
                         n_units_out=n_units_out,n_units_r=n_units_r, n_units_out_t=n_units_out_t,
                         n_units_r_t=n_units_r_t, penalty_l2=penalty_l2, penalty_l2_t=penalty_l2_t,
                         step_size=step_size, step_size_t=step_size_t, n_iter=n_iter,
                         batch_size=batch_size, n_iter_min=n_iter_min,
                         val_split_prop=val_split_prop, early_stopping=early_stopping,
                         patience=patience, verbose=verbose, n_iter_print=n_iter_print,
                         seed=seed, penalty_orthogonal=penalty_orthogonal,
                         n_units_r_small=n_units_r_small, nonlin=nonlin,
                         rescale_transformation=rescale_transformation)


class PWNet(PseudoOutcomeNet):
    """ Wrapper for PW-learner using PseudoOutcomeNet"""
    def __init__(self, first_stage_strategy: str = T_STRATEGY,
                 data_split: bool = False,
                 cross_fit: bool = False, n_cf_folds: int = DEFAULT_CF_FOLDS,
                 binary_y: bool = False,
                 n_layers_out: int = DEFAULT_LAYERS_OUT,
                 n_layers_r: int = DEFAULT_LAYERS_R,
                 n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
                 n_layers_r_t: int = DEFAULT_LAYERS_R_T,
                 n_units_out: int = DEFAULT_UNITS_OUT,
                 n_units_r: int = DEFAULT_UNITS_R,
                 n_units_out_t: int = DEFAULT_UNITS_OUT_T,
                 n_units_r_t: int = DEFAULT_UNITS_R_T,
                 penalty_l2: float = DEFAULT_PENALTY_L2,
                 penalty_l2_t: float = DEFAULT_PENALTY_L2,
                 step_size: float = DEFAULT_STEP_SIZE,
                 step_size_t: float = DEFAULT_STEP_SIZE_T,
                 n_iter: int = DEFAULT_N_ITER,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 n_iter_min: int = DEFAULT_N_ITER_MIN,
                 val_split_prop: float = DEFAULT_VAL_SPLIT,
                 early_stopping: bool = True,
                 patience: int = DEFAULT_PATIENCE,
                 verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                 seed: int = DEFAULT_SEED,
                 rescale_transformation: bool = False,
                 penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                 n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                 nonlin: str = DEFAULT_NONLIN
                 ):
        super().__init__(first_stage_strategy=first_stage_strategy, data_split=data_split,
                         cross_fit=cross_fit, n_cf_folds=n_cf_folds,
                         transformation=PW_TRANSFORMATION, binary_y=binary_y,
                         n_layers_out=n_layers_out, n_layers_r=n_layers_r,
                         n_layers_out_t=n_layers_out_t, n_layers_r_t=n_layers_r_t,
                         n_units_out=n_units_out,n_units_r=n_units_r, n_units_out_t=n_units_out_t,
                         n_units_r_t=n_units_r_t, penalty_l2=penalty_l2, penalty_l2_t=penalty_l2_t,
                         step_size=step_size, step_size_t=step_size_t, n_iter=n_iter,
                         batch_size=batch_size, n_iter_min=n_iter_min,
                         val_split_prop=val_split_prop, early_stopping=early_stopping,
                         patience=patience, verbose=verbose, n_iter_print=n_iter_print,
                         seed=seed, penalty_orthogonal=penalty_orthogonal,
                         n_units_r_small=n_units_r_small, nonlin=nonlin,
                         rescale_transformation=rescale_transformation)


def train_pseudooutcome_net(X, y, w, p=None, first_stage_strategy: str = T_STRATEGY,
                            data_split: bool = False,
                            cross_fit: bool = False, n_cf_folds: int = DEFAULT_CF_FOLDS,
                            transformation: str = DR_TRANSFORMATION,
                            binary_y: bool = False,
                            n_layers_out: int = DEFAULT_LAYERS_OUT,
                            n_layers_r: int = DEFAULT_LAYERS_R,
                            n_layers_r_t: int = DEFAULT_LAYERS_R_T,
                            n_layers_out_t: int = DEFAULT_LAYERS_OUT_T,
                            n_units_out: int = DEFAULT_UNITS_OUT,
                            n_units_r: int = DEFAULT_UNITS_R,
                            n_units_out_t: int = DEFAULT_UNITS_OUT_T,
                            n_units_r_t: int = DEFAULT_UNITS_R_T,
                            penalty_l2: float = DEFAULT_PENALTY_L2,
                            penalty_l2_t: float = DEFAULT_PENALTY_L2,
                            step_size: float = DEFAULT_STEP_SIZE,
                            step_size_t: float = DEFAULT_STEP_SIZE_T,
                            n_iter: int = DEFAULT_N_ITER,
                            batch_size: int = DEFAULT_BATCH_SIZE,
                            val_split_prop: float = DEFAULT_VAL_SPLIT,
                            early_stopping: bool = True,
                            patience: int = DEFAULT_PATIENCE,
                            n_iter_min: int = DEFAULT_N_ITER_MIN,
                            verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                            seed: int = DEFAULT_SEED, rescale_transformation: bool = False,
                            return_val_loss: bool = False,
                            penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                            n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                            nonlin: str = DEFAULT_NONLIN, avg_objective: bool = DEFAULT_AVG_OBJECTIVE):
    # get shape of data
    n, d = X.shape

    if p is not None:
        p = check_shape_1d_data(p)

    # get transformation function
    transformation_function = _get_transformation_function(transformation)

    # get strategy name
    if first_stage_strategy not in ALL_STRATEGIES:
        raise ValueError('Parameter first stage should be in '
                         'catenets.models.twostep_nets.ALL_STRATEGIES. '
                         'You passed {}'.format(first_stage_strategy))

    # split data as wanted
    if p is None or transformation is not PW_TRANSFORMATION:
        if not cross_fit:
            if not data_split:
                if verbose > 0:
                    print('Training first stage with all data (no data splitting)')
                # use all data for both
                fit_mask = onp.ones(n, dtype=bool)
                pred_mask = onp.ones(n, dtype=bool)
            else:
                if verbose > 0:
                    print('Training first stage with half of the data (data splitting)')
                # split data in half
                fit_idx = onp.random.choice(n, int(onp.round(n / 2)))
                fit_mask = onp.zeros(n, dtype=bool)

                fit_mask[fit_idx] = 1
                pred_mask = ~ fit_mask

            mu_0, mu_1, pi_hat = _train_and_predict_first_stage(X, y, w, fit_mask, pred_mask,
                                                                first_stage_strategy=first_stage_strategy,
                                                                binary_y=binary_y,
                                                                n_layers_out=n_layers_out,
                                                                n_layers_r=n_layers_r,
                                                                n_units_out=n_units_out,
                                                                n_units_r=n_units_r,
                                                                penalty_l2=penalty_l2,
                                                                step_size=step_size,
                                                                n_iter=n_iter,
                                                                batch_size=batch_size,
                                                                val_split_prop=val_split_prop,
                                                                early_stopping=early_stopping,
                                                                patience=patience,
                                                                n_iter_min=n_iter_min,
                                                                verbose=verbose,
                                                                n_iter_print=n_iter_print,
                                                                seed=seed,
                                                                penalty_orthogonal=penalty_orthogonal,
                                                                n_units_r_small=n_units_r_small,
                                                                nonlin=nonlin,
                                                                avg_objective=avg_objective,
                                                                transformation=transformation)
            if data_split:
                # keep only prediction data
                X, y, w = X[pred_mask, :], y[pred_mask, :], w[pred_mask, :]

                if p is not None:
                    p = p[pred_mask, :]

        else:
            if verbose > 0:
                print('Training first stage in {} folds (cross-fitting)'.format(n_cf_folds))
            # do cross fitting
            mu_0, mu_1, pi_hat = onp.zeros((n, 1)), onp.zeros((n, 1)), onp.zeros((n, 1))
            splitter = StratifiedKFold(n_splits=n_cf_folds, shuffle=True,
                                       random_state=seed)

            fold_count = 1
            for train_idx, test_idx in splitter.split(X, w):

                if verbose > 0:
                    print('Training fold {}.'.format(fold_count))
                fold_count = fold_count + 1

                pred_mask = onp.zeros(n, dtype=bool)
                pred_mask[test_idx] = 1
                fit_mask = ~ pred_mask

                mu_0[pred_mask], mu_1[pred_mask], pi_hat[pred_mask] = \
                    _train_and_predict_first_stage(X, y, w, fit_mask, pred_mask,
                                                   first_stage_strategy=first_stage_strategy,
                                                   binary_y=binary_y,
                                                   n_layers_out=n_layers_out,
                                                   n_layers_r=n_layers_r,
                                                   n_units_out=n_units_out,
                                                   n_units_r=n_units_r,
                                                   penalty_l2=penalty_l2,
                                                   step_size=step_size,
                                                   n_iter=n_iter,
                                                   batch_size=batch_size,
                                                   val_split_prop=val_split_prop,
                                                   early_stopping=early_stopping,
                                                   patience=patience,
                                                   n_iter_min=n_iter_min,
                                                   verbose=verbose,
                                                   n_iter_print=n_iter_print,
                                                   seed=seed,
                                                   penalty_orthogonal=penalty_orthogonal,
                                                   n_units_r_small=n_units_r_small,
                                                   nonlin=nonlin, avg_objective=avg_objective,
                                                   transformation=transformation)

    if verbose > 0:
        print('Training second stage.')

    if p is not None:
        # use known propensity score
        p = check_shape_1d_data(p)
        pi_hat = p

    # second stage
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    # transform data and fit on transformed data
    if transformation is PW_TRANSFORMATION:
        mu_0 = None
        mu_1 = None

    pseudo_outcome = transformation_function(y=y, w=w, p=pi_hat, mu_0=mu_0, mu_1=mu_1)
    if rescale_transformation:
        scale_factor = onp.std(y) / onp.std(pseudo_outcome)
        if scale_factor > 1:
            scale_factor = 1
        else:
            pseudo_outcome = scale_factor * pseudo_outcome
        params, predict_funs = train_output_net_only(X, pseudo_outcome, binary_y=False,
                                                     n_layers_out=n_layers_out_t,
                                                     n_units_out=n_units_out_t,
                                                     n_layers_r=n_layers_r_t,
                                                     n_units_r=n_units_r_t,
                                                     penalty_l2=penalty_l2_t,
                                                     step_size=step_size_t,
                                                     n_iter=n_iter,
                                                     batch_size=batch_size,
                                                     val_split_prop=val_split_prop,
                                                     early_stopping=early_stopping,
                                                     patience=patience,
                                                     n_iter_min=n_iter_min,
                                                     n_iter_print=n_iter_print,
                                                     verbose=verbose,
                                                     seed=seed,
                                                     return_val_loss=return_val_loss,
                                                     nonlin=nonlin,
                                                     avg_objective=avg_objective)
        return params, predict_funs, scale_factor
    else:
        return train_output_net_only(X, pseudo_outcome, binary_y=False,
                                     n_layers_out=n_layers_out_t,
                                     n_units_out=n_units_out_t,
                                     n_layers_r=n_layers_r_t,
                                     n_units_r=n_units_r_t,
                                     penalty_l2=penalty_l2_t,
                                     step_size=step_size_t,
                                     n_iter=n_iter,
                                     batch_size=batch_size,
                                     val_split_prop=val_split_prop,
                                     early_stopping=early_stopping,
                                     patience=patience,
                                     n_iter_min=n_iter_min,
                                     n_iter_print=n_iter_print,
                                     verbose=verbose,
                                     seed=seed,
                                     return_val_loss=return_val_loss, nonlin=nonlin,
                                     avg_objective=avg_objective)


def _train_and_predict_first_stage_t(X, y, w, fit_mask, pred_mask,
                                     binary_y: bool = False,
                                     n_layers_out: int = DEFAULT_LAYERS_OUT,
                                     n_units_out: int = DEFAULT_UNITS_OUT,
                                     n_layers_r: int = DEFAULT_LAYERS_R,
                                     n_units_r: int = DEFAULT_UNITS_R,
                                     penalty_l2: float = DEFAULT_PENALTY_L2,
                                     step_size: float = DEFAULT_STEP_SIZE,
                                     n_iter: int = DEFAULT_N_ITER,
                                     batch_size: int = DEFAULT_BATCH_SIZE,
                                     val_split_prop: float = DEFAULT_VAL_SPLIT,
                                     early_stopping: bool = True,
                                     patience: int = DEFAULT_PATIENCE,
                                     n_iter_min: int = DEFAULT_N_ITER_MIN,
                                     verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                                     seed: int = DEFAULT_SEED, nonlin: str = DEFAULT_NONLIN,
                                     avg_objective: bool = False,
                                     transformation: str = DR_TRANSFORMATION):
    # train and predict first stage estimators using TNet
    if len(w.shape) > 1:
        w = w.reshape((len(w),))

    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    if transformation is not PW_TRANSFORMATION:
        if verbose > 0:
            print('Training PO_0 Net')
        params_0, predict_fun_0 = train_output_net_only(X_fit[w_fit == 0], y_fit[w_fit == 0],
                                                        binary_y=binary_y,
                                                        n_layers_out=n_layers_out,
                                                        n_units_out=n_units_out,
                                                        n_layers_r=n_layers_r,
                                                        n_units_r=n_units_r,
                                                        penalty_l2=penalty_l2,
                                                        step_size=step_size,
                                                        n_iter=n_iter,
                                                        batch_size=batch_size,
                                                        val_split_prop=val_split_prop,
                                                        early_stopping=early_stopping,
                                                        patience=patience,
                                                        n_iter_min=n_iter_min,
                                                        n_iter_print=n_iter_print,
                                                        verbose=verbose,
                                                        seed=seed, nonlin=nonlin,
                                                        avg_objective=avg_objective)
        mu_0 = predict_fun_0(params_0, X_pred)

        if verbose > 0:
            print('Training PO_1 Net')
        params_1, predict_fun_1 = train_output_net_only(X_fit[w_fit == 1], y_fit[w_fit == 1],
                                                        binary_y=binary_y,
                                                        n_layers_out=n_layers_out,
                                                        n_units_out=n_units_out,
                                                        n_layers_r=n_layers_r,
                                                        n_units_r=n_units_r,
                                                        penalty_l2=penalty_l2,
                                                        step_size=step_size,
                                                        n_iter=n_iter,
                                                        batch_size=batch_size,
                                                        val_split_prop=val_split_prop,
                                                        early_stopping=early_stopping,
                                                        patience=patience,
                                                        n_iter_min=n_iter_min,
                                                        n_iter_print=n_iter_print,
                                                        verbose=verbose,
                                                        seed=seed, nonlin=nonlin,
                                                        avg_objective=avg_objective)
        mu_1 = predict_fun_1(params_1, X_pred)
    else:
        mu_0, mu_1 = onp.nan, onp.nan

    if transformation is not RA_TRANSFORMATION:
        if verbose > 0:
            print('Training propensity net')
        params_prop, predict_fun_prop = train_output_net_only(X_fit, w_fit,
                                                              binary_y=True,
                                                              n_layers_out=n_layers_out,
                                                              n_units_out=n_units_out,
                                                              n_layers_r=n_layers_r,
                                                              n_units_r=n_units_r,
                                                              penalty_l2=penalty_l2,
                                                              step_size=step_size,
                                                              n_iter=n_iter,
                                                              batch_size=batch_size,
                                                              val_split_prop=val_split_prop,
                                                              early_stopping=early_stopping,
                                                              patience=patience,
                                                              n_iter_min=n_iter_min,
                                                              n_iter_print=n_iter_print,
                                                              verbose=verbose,
                                                              seed=seed, nonlin=nonlin,
                                                              avg_objective=avg_objective)
        pi_hat = predict_fun_prop(params_prop, X_pred)
    else:
        pi_hat = onp.nan

    return mu_0, mu_1, pi_hat


def _train_and_predict_first_stage_s1(X, y, w, fit_mask, pred_mask, binary_y: bool = False,
                                      n_layers_out: int = DEFAULT_LAYERS_OUT,
                                      n_layers_r: int = DEFAULT_LAYERS_R,
                                      n_units_out: int = DEFAULT_UNITS_OUT,
                                      n_units_r: int = DEFAULT_UNITS_R,
                                      penalty_l2: float = DEFAULT_PENALTY_L2,
                                      step_size: float = DEFAULT_STEP_SIZE,
                                      n_iter: int = DEFAULT_N_ITER,
                                      batch_size: int = DEFAULT_BATCH_SIZE,
                                      val_split_prop: float = DEFAULT_VAL_SPLIT,
                                      early_stopping: bool = True,
                                      patience: int = DEFAULT_PATIENCE,
                                      n_iter_min: int = DEFAULT_N_ITER_MIN,
                                      verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                                      seed: int = DEFAULT_SEED, nonlin: str = DEFAULT_NONLIN,
                                      avg_objective: bool = False,
                                      transformation: str = DR_TRANSFORMATION):
    # Train and predict first stage estimators using SNet1/ TARNet
    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    if verbose > 0:
        print('Training SNet1')
    params_cfr, predict_funs_cfr = train_snet1(X_fit, y_fit, w_fit, binary_y=binary_y,
                                               n_layers_r=n_layers_r,
                                               n_units_r=n_units_r, n_layers_out=n_layers_out,
                                               n_units_out=n_units_out, penalty_l2=penalty_l2,
                                               penalty_disc=0, step_size=step_size,
                                               n_iter=n_iter, batch_size=batch_size,
                                               val_split_prop=val_split_prop,
                                               early_stopping=early_stopping,
                                               patience=patience, n_iter_min=n_iter_min,
                                               verbose=verbose, n_iter_print=n_iter_print,
                                               seed=seed, nonlin=nonlin,
                                               avg_objective=avg_objective)
    _, mu_0_hat, mu_1_hat = predict_snet1(X_pred, params_cfr, predict_funs_cfr, return_po=True)

    if transformation is not RA_TRANSFORMATION:
        if verbose > 0:
            print('Training propensity net')
        params_prop, predict_fun_prop = train_output_net_only(X_fit, w_fit,
                                                              binary_y=True,
                                                              n_layers_out=n_layers_out,
                                                              n_units_out=n_units_out,
                                                              n_layers_r=n_layers_r,
                                                              n_units_r=n_units_r,
                                                              penalty_l2=penalty_l2,
                                                              step_size=step_size,
                                                              n_iter=n_iter,
                                                              batch_size=batch_size,
                                                              val_split_prop=val_split_prop,
                                                              early_stopping=early_stopping,
                                                              patience=patience,
                                                              n_iter_min=n_iter_min,
                                                              n_iter_print=n_iter_print,
                                                              verbose=verbose,
                                                              seed=seed, nonlin=nonlin,
                                                              avg_objective=avg_objective)
        pi_hat = predict_fun_prop(params_prop, X_pred)
    else:
        pi_hat = onp.nan

    return mu_0_hat, mu_1_hat, pi_hat


def _train_and_predict_first_stage_s2(X, y, w, fit_mask, pred_mask, binary_y: bool = False,
                                      n_layers_out: int = DEFAULT_LAYERS_OUT,
                                      n_layers_r: int = DEFAULT_LAYERS_R,
                                      n_units_out: int = DEFAULT_UNITS_OUT,
                                      n_units_r: int = DEFAULT_UNITS_R,
                                      penalty_l2: float = DEFAULT_PENALTY_L2,
                                      step_size: float = DEFAULT_STEP_SIZE,
                                      n_iter: int = DEFAULT_N_ITER,
                                      batch_size: int = DEFAULT_BATCH_SIZE,
                                      val_split_prop: float = DEFAULT_VAL_SPLIT,
                                      early_stopping: bool = True,
                                      patience: int = DEFAULT_PATIENCE,
                                      n_iter_min: int = DEFAULT_N_ITER_MIN,
                                      verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                                      seed: int = DEFAULT_SEED, nonlin: str = DEFAULT_NONLIN,
                                      avg_objective: bool = False):
    # train and predict first stage estimator using SNet2/DragonNet
    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    if verbose > 0:
        print('Training SNet2')
    params, predict_funs = train_snet2(X_fit, y_fit, w_fit, binary_y=binary_y,
                                       n_layers_r=n_layers_r, n_units_r=n_units_r,
                                       n_layers_out=n_layers_out, n_units_out=n_units_out,
                                       penalty_l2=penalty_l2, step_size=step_size,
                                       n_iter=n_iter, batch_size=batch_size,
                                       val_split_prop=val_split_prop,
                                       early_stopping=early_stopping,
                                       patience=patience, n_iter_min=n_iter_min,
                                       verbose=verbose, n_iter_print=n_iter_print,
                                       seed=seed, nonlin=nonlin, avg_objective=avg_objective)

    _, mu_0_hat, mu_1_hat, pi_hat = predict_snet2(X_pred, params, predict_funs, return_po=True,
                                                  return_prop=True)
    return mu_0_hat, mu_1_hat, pi_hat


def _train_and_predict_first_stage_s3(X, y, w, fit_mask, pred_mask, binary_y: bool = False,
                                      n_layers_out: int = DEFAULT_LAYERS_OUT,
                                      n_layers_r: int = DEFAULT_LAYERS_R,
                                      n_units_out: int = DEFAULT_UNITS_OUT,
                                      n_units_r: int = DEFAULT_UNITS_R_BIG_S3,
                                      penalty_l2: float = DEFAULT_PENALTY_L2,
                                      step_size: float = DEFAULT_STEP_SIZE,
                                      n_iter: int = DEFAULT_N_ITER,
                                      batch_size: int = DEFAULT_BATCH_SIZE,
                                      val_split_prop: float = DEFAULT_VAL_SPLIT,
                                      early_stopping: bool = True,
                                      patience: int = DEFAULT_PATIENCE,
                                      n_iter_min: int = DEFAULT_N_ITER_MIN,
                                      verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                                      seed: int = DEFAULT_SEED,
                                      penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                                      n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S3,
                                      nonlin: str = DEFAULT_NONLIN, avg_objective: bool = False):
    # Train and predict first stage estimator using SNet3/DR-CFR
    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    # use snet3
    if verbose > 0:
        print('Training SNet3')
    params, predict_funs = train_snet3(X_fit, y_fit, w_fit, binary_y=binary_y,
                                       n_layers_r=n_layers_r, n_units_r=n_units_r,
                                       n_layers_out=n_layers_out, n_units_out=n_units_out,
                                       penalty_l2=penalty_l2, step_size=step_size,
                                       n_iter=n_iter, batch_size=batch_size,
                                       val_split_prop=val_split_prop,
                                       early_stopping=early_stopping,
                                       patience=patience, n_iter_min=n_iter_min,
                                       verbose=verbose, n_iter_print=n_iter_print,
                                       seed=seed, penalty_orthogonal=penalty_orthogonal,
                                       n_units_r_small=n_units_r_small, nonlin=nonlin,
                                       avg_objective=avg_objective)

    _, mu_0_hat, mu_1_hat, pi_hat = predict_snet3(X_pred, params, predict_funs, return_po=True,
                                                  return_prop=True)
    return mu_0_hat, mu_1_hat, pi_hat


def _train_and_predict_first_stage_s4(X, y, w, fit_mask, pred_mask, binary_y: bool = False,
                                      n_layers_out: int = DEFAULT_LAYERS_OUT,
                                      n_layers_r: int = DEFAULT_LAYERS_R,
                                      n_units_out: int = DEFAULT_UNITS_OUT,
                                      n_units_r: int = DEFAULT_UNITS_R_BIG_S,
                                      penalty_l2: float = DEFAULT_PENALTY_L2,
                                      step_size: float = DEFAULT_STEP_SIZE,
                                      n_iter: int = DEFAULT_N_ITER,
                                      batch_size: int = DEFAULT_BATCH_SIZE,
                                      val_split_prop: float = DEFAULT_VAL_SPLIT,
                                      early_stopping: bool = True,
                                      patience: int = DEFAULT_PATIENCE,
                                      n_iter_min: int = DEFAULT_N_ITER_MIN,
                                      verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                                      seed: int = DEFAULT_SEED,
                                      penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                                      n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                                      nonlin: str = DEFAULT_NONLIN, avg_objective: bool = False):
    # train and predict first stage using SNet
    # split the data
    X_fit, y_fit, w_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]
    X_pred = X[pred_mask, :]

    if verbose > 0:
        print('Training SNet')
    params, predict_funs = train_snet(X_fit, y_fit, w_fit, binary_y=binary_y,
                                      n_layers_r=n_layers_r, n_units_r=n_units_r,
                                      n_layers_out=n_layers_out, n_units_out=n_units_out,
                                      penalty_l2=penalty_l2, step_size=step_size,
                                      n_iter=n_iter, batch_size=batch_size,
                                      val_split_prop=val_split_prop,
                                      early_stopping=early_stopping,
                                      patience=patience, n_iter_min=n_iter_min,
                                      verbose=verbose, n_iter_print=n_iter_print,
                                      seed=seed, penalty_orthogonal=penalty_orthogonal,
                                      n_units_r_small=n_units_r_small, nonlin=nonlin,
                                      avg_objective=avg_objective)

    _, mu_0_hat, mu_1_hat, pi_hat = predict_snet(X_pred, params, predict_funs, return_po=True,
                                                 return_prop=True)
    return mu_0_hat, mu_1_hat, pi_hat


def _train_and_predict_first_stage(X, y, w, fit_mask, pred_mask, first_stage_strategy,
                                   binary_y: bool = False, n_layers_out: int = DEFAULT_LAYERS_OUT,
                                   n_layers_r: int = DEFAULT_LAYERS_R,
                                   n_units_out: int = DEFAULT_UNITS_OUT,
                                   n_units_r: int = DEFAULT_UNITS_R,
                                   penalty_l2: float = DEFAULT_PENALTY_L2,
                                   step_size: float = DEFAULT_STEP_SIZE,
                                   n_iter: int = DEFAULT_N_ITER,
                                   batch_size: int = DEFAULT_BATCH_SIZE,
                                   val_split_prop: float = DEFAULT_VAL_SPLIT,
                                   early_stopping: bool = True,
                                   patience: int = DEFAULT_PATIENCE,
                                   n_iter_min: int = DEFAULT_N_ITER_MIN,
                                   verbose: int = 1, n_iter_print: int = DEFAULT_N_ITER_PRINT,
                                   seed: int = DEFAULT_SEED,
                                   penalty_orthogonal: float = DEFAULT_PENALTY_ORTHOGONAL,
                                   n_units_r_small: int = DEFAULT_UNITS_R_SMALL_S,
                                   nonlin: str = DEFAULT_NONLIN, avg_objective: bool = False,
                                   transformation: str = DR_TRANSFORMATION):
    if first_stage_strategy == T_STRATEGY:
        # simplest case: train three seperate heads
        mu_0, mu_1, pi_hat = _train_and_predict_first_stage_t(X, y, w, fit_mask, pred_mask,
                                                              binary_y=binary_y,
                                                              n_layers_out=n_layers_out,
                                                              n_units_out=n_units_out,
                                                              n_layers_r=n_layers_r,
                                                              n_units_r=n_units_r,
                                                              penalty_l2=penalty_l2,
                                                              step_size=step_size,
                                                              n_iter=n_iter,
                                                              batch_size=batch_size,
                                                              val_split_prop=val_split_prop,
                                                              early_stopping=early_stopping,
                                                              patience=patience,
                                                              n_iter_min=n_iter_min,
                                                              n_iter_print=n_iter_print,
                                                              verbose=verbose,
                                                              seed=seed, nonlin=nonlin,
                                                              avg_objective=avg_objective,
                                                              transformation=transformation)
    elif first_stage_strategy == S1_STRATEGY:
        # train TARNET and a seperate propensity head
        mu_0, mu_1, pi_hat = _train_and_predict_first_stage_s1(X, y, w, fit_mask, pred_mask,
                                                               binary_y=binary_y,
                                                               n_layers_out=n_layers_out,
                                                               n_layers_r=n_layers_r,
                                                               n_units_out=n_units_out,
                                                               n_units_r=n_units_r,
                                                               penalty_l2=penalty_l2,
                                                               step_size=step_size,
                                                               n_iter=n_iter,
                                                               batch_size=batch_size,
                                                               val_split_prop=val_split_prop,
                                                               early_stopping=early_stopping,
                                                               patience=patience,
                                                               n_iter_min=n_iter_min,
                                                               verbose=verbose,
                                                               n_iter_print=n_iter_print,
                                                               seed=seed, nonlin=nonlin,
                                                               avg_objective=avg_objective,
                                                               transformation=transformation)
    elif first_stage_strategy == S2_STRATEGY:
        # train snet2
        mu_0, mu_1, pi_hat = _train_and_predict_first_stage_s2(X, y, w, fit_mask, pred_mask,
                                                               binary_y=binary_y,
                                                               n_layers_out=n_layers_out,
                                                               n_layers_r=n_layers_r,
                                                               n_units_out=n_units_out,
                                                               n_units_r=n_units_r,
                                                               penalty_l2=penalty_l2,
                                                               step_size=step_size,
                                                               n_iter=n_iter,
                                                               batch_size=batch_size,
                                                               val_split_prop=val_split_prop,
                                                               early_stopping=early_stopping,
                                                               patience=patience,
                                                               n_iter_min=n_iter_min,
                                                               verbose=verbose,
                                                               n_iter_print=n_iter_print,
                                                               seed=seed, nonlin=nonlin,
                                                               avg_objective=avg_objective)
    elif first_stage_strategy == S3_STRATEGY:
        mu_0, mu_1, pi_hat = _train_and_predict_first_stage_s3(X, y, w, fit_mask, pred_mask,
                                                               binary_y=binary_y,
                                                               n_layers_out=n_layers_out,
                                                               n_layers_r=n_layers_r,
                                                               n_units_out=n_units_out,
                                                               n_units_r=n_units_r,
                                                               penalty_l2=penalty_l2,
                                                               step_size=step_size,
                                                               n_iter=n_iter,
                                                               batch_size=batch_size,
                                                               val_split_prop=val_split_prop,
                                                               early_stopping=early_stopping,
                                                               patience=patience,
                                                               n_iter_min=n_iter_min,
                                                               verbose=verbose,
                                                               n_iter_print=n_iter_print,
                                                               seed=seed,
                                                               penalty_orthogonal=penalty_orthogonal,
                                                               n_units_r_small=n_units_r_small,
                                                               nonlin=nonlin,
                                                               avg_objective=avg_objective)
    else:
        mu_0, mu_1, pi_hat = _train_and_predict_first_stage_s4(X, y, w, fit_mask, pred_mask,
                                                               binary_y=binary_y,
                                                               n_layers_out=n_layers_out,
                                                               n_layers_r=n_layers_r,
                                                               n_units_out=n_units_out,
                                                               n_units_r=n_units_r,
                                                               penalty_l2=penalty_l2,
                                                               step_size=step_size,
                                                               n_iter=n_iter,
                                                               batch_size=batch_size,
                                                               val_split_prop=val_split_prop,
                                                               early_stopping=early_stopping,
                                                               patience=patience,
                                                               n_iter_min=n_iter_min,
                                                               verbose=verbose,
                                                               n_iter_print=n_iter_print,
                                                               seed=seed,
                                                               penalty_orthogonal=penalty_orthogonal,
                                                               n_units_r_small=n_units_r_small,
                                                               nonlin=nonlin,
                                                               avg_objective=avg_objective)

    return mu_0, mu_1, pi_hat
