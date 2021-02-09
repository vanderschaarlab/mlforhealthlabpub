# Conditional Independence Testing with Generative Adversarial Networks

This is a python implementation of the algorithm in the paper ["Conditional Independence Testing with Generative Adversarial Networks"](https://arxiv.org/pdf/1907.04068.pdf). The goal of this project is to test conditional independence between variable sets X and Y conditional on Z. It contains synthetic data generation to understand the behaviour of our test in various settings. 

*Please cite the above paper if this resource is used in any publication*

## Dependencies
The only significant dependencies are python 3.6 or later. It is known to work with tensorflow 1.13.1.

## First steps
To get started, check *tutorial_gcit.ipynb* which will guide you through the test from the beginning. 

## Use case on genetic data
We include in the *CCLE Experiments* folder the code used in the real data experiment on Section 5 of the main body of this paper. The folder includes the data used and a simple script to test conditional independence of each feature and drug response given all other features.
