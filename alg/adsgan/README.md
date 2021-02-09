# Codebase for Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN)

Authors: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar

Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar, 
"Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
A harmonizing advancement for AI in medicine," 
IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
 
Paper Link: https://ieeexplore.ieee.org/document/9034117

Contact: jsyoon0823@gmail.com

This directory contains implementations of ADSGAN framework for synthetic data generation.

To run the pipeline for training and evaluation on ADSGAN framwork, simply run 
python3 -m main_adsgan.py.

### Code explanation

(1) data_loader.py
- Load and preprocess original data

(2) adsgan.py
- Generate synthetic data using the original data

(3) feature_distribution.py
- Compare feature distribution between original data and synthetic data

(4) compute_wd.py
- Compare Wasserstein distance between original data and synthetic data

(5) compute_identifiability.py
- Compare Identifiability between original data and synthetic data

(6) main_adsgan.py
- Report the performances of ADS-GAN in terms of Wasserstein Distance, Identifiability, and Distribution.

### Command inputs:

-   iterations: Number of experiments iterations
-   lamda: Hyper-parameter to control the identifiability and quality of the synthetic data
-   h_dim: Number of hidden state dimensions
-   z_dim: Number of random state dimensions
-   mb_size: Number of mini-batch samples

Note that hyper-parameters should be optimized for different datasets.

### Example command

```shell
$ python3 main_adsgan.py --iterations 10000 --lamda 0.1 --h_dim 30
--z_dim 10 --mb_size 128
```

### Outputs

-   orig_data: original data
-   synth_data: synthetically generated data
-   measures: performances of 3 metrics (Distribution, WD, identifiability)
