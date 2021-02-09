import torch
import torch.nn as nn
import logging
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_sample
from pyro.infer.predictive import _guess_max_plate_nesting
from pyro.nn.module import PyroModule
from pyro.optim import DCTAdam

from pyro.contrib.forecast.util import (MarkDCTParamMessenger, PrefixConditionMessenger, PrefixReplayMessenger, PrefixWarmStartMessenger,
                   reshape_batch)


logger = logging.getLogger(__name__)


class EnumForecaster(nn.Module):
    """
    Forecaster for a :class:`ForecastingModel` using variational inference.

    On initialization, this fits a distribution using variational inference
    over latent variables and exact inference over the noise distribution,
    typically a :class:`~pyro.distributions.GaussianHMM` or variant.

    After construction this can be called to generate sample forecasts.

    :ivar list losses: A list of losses recorded during training, typically
        used to debug convergence. Defined by ``loss = -elbo / data.numel()``.

    :param ForecastingModel model: A forecasting model subclass instance.
    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor

    :param guide: Optional guide instance. Defaults to a
        :class:`~pyro.infer.autoguide.AutoNormal`.
    :type guide: ~pyro.nn.module.PyroModule
    :param callable init_loc_fn: A per-site initialization function for the
        :class:`~pyro.infer.autoguide.AutoNormal` guide. Defaults to
        :func:`~pyro.infer.autoguide.initialization.init_to_sample`. See
        :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial uncertainty scale of the
        :class:`~pyro.infer.autoguide.AutoNormal` guide.
    :param callable create_plates: An optional function to create plates for
        subsampling with the :class:`~pyro.infer.autoguide.AutoNormal` guide.
    :param optim: An optional Pyro optimizer. Defaults to a freshly constructed
        :class:`~pyro.optim.optim.DCTAdam`.
    :type optim: ~pyro.optim.optim.PyroOptim
    :param float learning_rate: Learning rate used by
        :class:`~pyro.optim.optim.DCTAdam`.
    :param tuple betas: Coefficients for running averages used by
        :class:`~pyro.optim.optim.DCTAdam`.
    :param float learning_rate_decay: Learning rate decay used by
        :class:`~pyro.optim.optim.DCTAdam`. Note this is the total decay
        over all ``num_steps``, not the per-step decay factor.
    :param float clip_norm: Norm used for gradient clipping during
        optimization. Defaults to 10.0.
    :param bool dct_gradients: Whether to discrete cosine transform gradients
        in :class:`~pyro.optim.optim.DCTAdam`. Defaults to False.
    :param bool subsample_aware: whether to update gradient statistics only
        for those elements that appear in a subsample. This is used
        by :class:`~pyro.optim.optim.DCTAdam`.
    :param int num_steps: Number of :class:`~pyro.infer.svi.SVI` steps.
    :param int num_particles: Number of particles used to compute the
        :class:`~pyro.infer.elbo.ELBO`.
    :param bool vectorize_particles: If ``num_particles > 1``, determines
        whether to vectorize computation of the :class:`~pyro.infer.elbo.ELBO`.
        Defaults to True. Set to False for models with dynamic control flow.
    :param bool warm_start: Whether to warm start parameters from a smaller
        time window. Note this may introduce statistical leakage; usage is
        recommended for model exploration purposes only and should be disabled
        when publishing metrics.
    :param int log_every: Number of training steps between logging messages.
    """
    def __init__(self, model, data, covariates, *,
                 guide=None,
                 init_loc_fn=init_to_sample,
                 init_scale=0.1,
                 create_plates=None,
                 optim=None,
                 learning_rate=0.01,
                 betas=(0.9, 0.99),
                 learning_rate_decay=0.1,
                 clip_norm=10.0,
                 dct_gradients=False,
                 subsample_aware=False,
                 num_steps=1001,
                 num_particles=1,
                 vectorize_particles=True,
                 warm_start=False,
                 log_every=100):
        assert data.size(-2) == covariates.size(-2)
        super().__init__()
        self.model = model
        if guide is None:
            guide = AutoNormal(self.model, init_loc_fn=init_loc_fn, init_scale=init_scale,
                               create_plates=create_plates)
        self.guide = guide

        # Initialize.
        if warm_start:
            model = PrefixWarmStartMessenger()(model)
            guide = PrefixWarmStartMessenger()(guide)
        if dct_gradients:
            model = MarkDCTParamMessenger("time")(model)
            guide = MarkDCTParamMessenger("time")(guide)
        elbo = TraceEnum_ELBO(num_particles=num_particles,
                              vectorize_particles=vectorize_particles)
        elbo._guess_max_plate_nesting(model, guide, (data, covariates), {})
        elbo.max_plate_nesting = max(elbo.max_plate_nesting, 1)  # force a time plate

        losses = []
        if num_steps:
            if optim is None:
                optim = DCTAdam({"lr": learning_rate, "betas": betas,
                                 "lrd": learning_rate_decay ** (1 / num_steps),
                                 "clip_norm": clip_norm,
                                 "subsample_aware": subsample_aware})
            svi = SVI(self.model, self.guide, optim, elbo)
            for step in range(num_steps):
                loss = svi.step(data, covariates) / data.numel()
                if log_every and step % log_every == 0:
                    logger.info("step {: >4d} loss = {:0.6g}".format(step, loss))
                losses.append(loss)

        self.guide.create_plates = None  # Disable subsampling after training.
        self.max_plate_nesting = elbo.max_plate_nesting
        self.losses = losses

    def __call__(self, data, covariates, num_samples, batch_size=None):
        """
        Samples forecasted values of data for time steps in ``[t1,t2)``, where
        ``t1 = data.size(-2)`` is the duration of observed data and ``t2 =
        covariates.size(-2)`` is the extended duration of covariates. For
        example to forecast 7 days forward conditioned on 30 days of
        observations, set ``t1=30`` and ``t2=37``.

        :param data: A tensor dataset with time dimension -2.
        :type data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
            For models not using covariates, pass a shaped empty tensor
            ``torch.empty(duration, 0)``.
        :type covariates: ~torch.Tensor
        :param int num_samples: The number of samples to generate.
        :param int batch_size: Optional batch size for sampling. This is useful
            for generating many samples from models with large memory
            footprint. Defaults to ``num_samples``.
        :returns: A batch of joint posterior samples of shape
            ``(num_samples,1,...,1) + data.shape[:-2] + (t2-t1,data.size(-1))``,
            where the ``1``'s are inserted to avoid conflict with model plates.
        :rtype: ~torch.Tensor
        """
        return super().__call__(data, covariates, num_samples, batch_size)


    def forward(self, data, covariates, num_samples, batch_size=None):
        assert data.size(-2) < covariates.size(-2)
        assert isinstance(num_samples, int) and num_samples > 0
        if batch_size is not None:
            batches = []
            while num_samples > 0:
                batch = self.forward(data, covariates, min(num_samples, batch_size))
                batches.append(batch)
                num_samples -= batch_size
            return torch.cat(batches)

        assert self.max_plate_nesting >= 1
        dim = -1 - self.max_plate_nesting

        with torch.no_grad():
            with poutine.trace() as tr:
                with pyro.plate("particles", num_samples, dim=dim):
                    self.guide(data, covariates)
            with PrefixReplayMessenger(tr.trace):
                with PrefixConditionMessenger(self.model._prefix_condition_data):
                    with pyro.plate("particles", num_samples, dim=dim):
                        return self.model(data, covariates)
