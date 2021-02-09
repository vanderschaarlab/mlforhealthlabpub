# Codebase for "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"

Authors: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
 
Paper Link: TBD

Contact: jsyoon0823@gmail.com

This directory contains implementations of VIME framework for self- and semi-supervised learning to tabular domain
using MNIST dataset.

-   MNIST data: http://yann.lecun.com/exdb/mnist/

To run the pipeline for training and evaluation on VIME framwork, simply run 
python3 -m main_vime.py or see jupyter-notebook tutorial of TimeGAN in tutorial_vime.ipynb.

Note that any model architecture can be used as the encoder and 
predictor model such as CNNs. 

### Code explanation

(1) data_loader.py
- Load and preprocess MNIST data to make it as tabular data

(2) supervised_model.py
- Define logistic regression, MLP, and XGBoost models
- All of them are supervised model for classification

(3) vime_self.py
- Self-supervised learning part of VIME framework
- Return the encoder part of VIME framework

(4) vime_semi.py
- Semi-supervised learning part of VIME framework
- Save trained predictor and return the predictions on the testing set

(5) main_vime.py
- Report the prediction performances of supervised-learning, Self-supervised part of VIME and entire VIME frameworks.

(6) vime_utils.py
- Some utility functions for metrics and VIME.

### Command inputs:

-   iterations: Number of experiments iterations
-   label_no: Number of labeled data to be used
-   model_name: supervised model name (mlp, logit, or xgboost)
-   p_m: corruption probability for self-supervised learning
-   alpha: hyper-parameter to control the weights of feature and mask losses
-   K: number of augmented samples
-   beta: hyperparameter to control supervised and unsupervised loss
-   label_data_rate: ratio of labeled data

Note that hyper-parameters should be optimized for different datasets.

### Example command

```shell
$ python3 main_vime.py --iterations 10 --label_no 1000 --model_name xgboost
--p_m 0.3 --alpha 2.0 --K 3 --beta 1.0 --label_data_rate 0.1 
```

### Outputs

-   results: performances of 3 different models (supervised only, VIME-self, and VIME)