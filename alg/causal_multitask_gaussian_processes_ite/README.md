# Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes


[![Tests](https://github.com/vanderschaarlab/mlforhealthlabpub/actions/workflows/test_cmgp.yml/badge.svg)](https://github.com/vanderschaarlab/mlforhealthlabpub/actions/workflows/test_cmgp.yml)
[![Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://vanderschaarlab.slack.com/messages/general)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/LICENSE.md)


**Code Author**: Ahmed M. Alaa


**Paper**: Ahmed M. Alaa, Mihaela van der Schaar, "Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes", NIPS 2017.

### Description
Predicated on the increasing abundance of electronic health records, we investigate the problem of inferring individualized treatment effects using observational
data. Stemming from the potential outcomes model, we propose a novel multitask learning framework in which factual and counterfactual outcomes are modeled as the outputs of a function in a vector-valued reproducing kernel Hilbert space (vvRKHS). We develop a nonparametric Bayesian method for learning the treatment effects using a multi-task Gaussian process (GP) with a linear coregionalization kernel as a prior over the vvRKHS. The Bayesian approach allows us to compute individualized measures of confidence in our estimates via pointwise credible intervals, which are crucial for realizing the full potential of precision
medicine. The impact of selection bias is alleviated via a risk-based empirical Bayes method for adapting the multi-task GP prior, which jointly minimizes the
empirical error in factual outcomes and the uncertainty in (unobserved) counterfactual outcomes. We conduct experiments on observational datasets for an interventional social program applied to premature infants, and a left ventricular assist device applied to cardiac patients wait-listed for a heart transplant. In both experiments, we show that our method significantly outperforms the state-of-the-art.

### Installation

```bash
$ pip install cmgp
```



### Example Usage


```python
from cmgp import CMGP
from cmgp.datasets import load
from cmgp.utils.metrics import sqrt_PEHE_with_diff

X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("twins")
 
model = CMGP(X_train, W_train, Y_train, max_gp_iterations=100)

pred = model.predict(X_test)
 
pehe = sqrt_PEHE_with_diff(Y_test, pred)

print(f"PEHE score for CMGP on {dataset} = {pehe}")
```


## References
1. [Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes](https://papers.nips.cc/paper/6934-bayesian-inference-of-individualized-treatment-effects-using-multi-task-gaussian-processes.pdf)
2. [Limits of Estimating Heterogeneous Treatment Effects:Guidelines for Practical Algorithm Design](http://proceedings.mlr.press/v80/alaa18a/alaa18a.pdf)
3. J. L. Hill. Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 2012.
