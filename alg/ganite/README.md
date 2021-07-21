# GANITE - Estimation of individualized treatment effects using generative adversarial nets

[![Tests](https://github.com/vanderschaarlab/mlforhealthlabpub/actions/workflows/test_ganite.yml/badge.svg)](https://github.com/vanderschaarlab/mlforhealthlabpub/actions/workflows/test_ganite.yml)
[![Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://vanderschaarlab.slack.com/messages/general)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/LICENSE.md)


**Code Author**: Jinsung Yoon (jsyoon0823@g.ucla.edu)


**Paper**: Jinsung Yoon, James Jordon, Mihaela van der Schaar, "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
 International Conference on Learning Representations (ICLR), 2018.

### Description

Estimating individualized treatment effects (ITE) is a challenging task due to the need for an individual’s potential outcomes to be learned from biased data and
without having access to the counterfactuals. We propose a novel method for inferring ITE based on the Generative Adversarial Nets (GANs) framework. Our method, termed Generative Adversarial Nets for inference of Individualized Treatment Effects (GANITE), is motivated by the possibility that we can capture the
uncertainty in the counterfactual distributions by attempting to learn them using a GAN. We generate proxies of the counterfactual outcomes using a counterfactual
generator, G, and then pass these proxies to an ITE generator, I, in order to train it. By modeling both of these using the GAN framework, we are able to infer
based on the factual data, while still accounting for the unseen counterfactuals. We test our method on three real-world datasets (with both binary and multiple
treatments) and show that GANITE outperforms state-of-the-art methods.

### Installation

```bash
$ pip install ganite
```



### Example Usage


```python
from ganite import Ganite
from ganite.datasets import load
from ganite.utils.metrics import sqrt_PEHE_with_diff

X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("twins")
 
model = Ganite(X_train, W_train, Y_train, num_iterations=500)
 
pred = model.predict(X_test).to_numpy()
 
pehe = sqrt_PEHE_with_diff(Y_test, pred)

print(f"PEHE score for GANITE on {dataset} = {pehe}")
```
