2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

1. Datasets (in Data Folder)
(1) Google dataset
- GOOGLE_BIG.csv
(2) Sine dataset
- Generated from data_loading.py 

2. Codes
(1) data_loading.py
- Transform raw time-series data to preprocessed time-series data (Googld data)
- Generate Sine data

(2) Metrics Folder
  (a) visualization_metrics.py
  - PCA and t-SNE analysis between Original data and Synthetic data
  (b) discriminative_score_metrics.py
  - Use Post-hoc RNN to classify Original data and Synthetic data
  (c) predictive_score_metrics.py
  - Use Post-hoc RNN to predict one-step ahead (last feature)

(2) tgan.py
- Use original time-series data as training set to generater synthetic time-series data

(3) main.py
- Replicate the performances of Table 2 and Figure 3 in the paper
- Report discriminative and predictive scores for each dataset and t-SNE and PCA analysis

3. How to use?
(1) In order to replicate the results in the paper
- Run main.py (with selecting the dataset)

(2) In order to achieve time-series synthetic dataset (Main objective of the paper)
- Run tgan.py
- Input original time-series data to achieve corresponding synthetic time-series data