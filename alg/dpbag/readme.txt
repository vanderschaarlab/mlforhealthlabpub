2019 NeurIPS Submission
Title: Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate
Authors: James Jordon, Jinsung Yoon, Mihaela van der Schaar

Last Updated Date: May 28th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

1. Dataset:
UCI Adult data
- adult.data
- adult.test

2. Codes
(1) data_loading.py
- Transform raw data to preprocessed data (train, test, valid sets)

(2) DPBag_Final.py
- Core Differentially Private Bagging algorithm
- Use train and valid sets with user-defined parameters (n, k, epsilon, delta) 
  to make Differentially private classification model  
- Use testset to evaluate the performance of Differentially private classification model

(3) main.py
- Replicate the performances of Table 1 and 2 in the paper
- Report Accuracy, AUROC, AUPRC, and Privacy Budget for each dataset and each differential privacy inputs

3. How to use?
(1) In order to replicate the results in the paper
- Run main.py with user-defined parameters (n, k, epsilon, delta)

(2) In order to achieve Differentially private classification model (Main objective of the paper)
- Use DPBag_Final.py
- Input train set and valid set with user-defined parameters (n, k, epsilon, delta) 
  to achieve Differentially private classification model (Last output)
