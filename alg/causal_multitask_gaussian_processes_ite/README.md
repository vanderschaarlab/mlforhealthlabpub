# Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes

## Dependencies

This implementation requires the IHDP dataset introduced in [3].

## Usage

```
python test_models.py -n "number of experiments" -t "test data ratio" -m "mode" [ -o <result.json> ]
```

The argument "mode" can be set to "NSGP" or "CMGP".

## References
1. [Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes](https://papers.nips.cc/paper/6934-bayesian-inference-of-individualized-treatment-effects-using-multi-task-gaussian-processes.pdf)
2. [Limits of Estimating Heterogeneous Treatment Effects:Guidelines for Practical Algorithm Design](http://proceedings.mlr.press/v80/alaa18a/alaa18a.pdf)
3. J. L. Hill. Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 2012.
