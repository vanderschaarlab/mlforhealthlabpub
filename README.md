# van der Schaar Lab
This repository contains the implementations of algorithms developed
by the [van der Schaar Lab](https://www.vanderschaar-lab.com/).

Please send comments and suggestions to [nm736@cam.ac.uk](mailto:nm736@cam.ac.uk)

## Content
An overview of the content of this repository is as below:
```python
.
├── alg/        # Directory contains algorithms.
├── app/        # Directory contains apps.
├── cfg/        # Directory contains common config.
├── doc/        # Directory contains common docs.
├── init/       # Directory contains algorithms.
├── template/   # Directory contains templates.
└── util/       # Directory contains common utilities.
```

## Publications
* The publications and the corresponding locations in the repo are listed below:

Paper [[Link]](#) | Journal/Conference | Code
--- | --- | ---
Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes [[Link]](https://proceedings.neurips.cc/paper/2017/hash/6a508a60aa3bf9510ea6acb021c94b48-Abstract.html) | NIPS 2017 | [alg/causal_multitask_gaussian_processes_ite](alg/causal_multitask_gaussian_processes_ite)
Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks [[Link]](https://proceedings.neurips.cc/paper/2017/hash/861dc9bd7f4e7dd3cccd534d0ae2a2e9-Abstract.html) | NIPS 2017 | [alg/dgp_survival](alg/dgp_survival)
AutoPrognosis: Automated Clinical Prognostic Modeling via Bayesian Optimization with Structured Kernel Learning [[Link]](https://icml.cc/Conferences/2018/Schedule?showEvent=2050) | ICML 2018 | [alg/autoprognosis/](alg/autoprognosis/)
Limits of Estimating Heterogeneous Treatment Effects: Guidelines for Practical Algorithm Design [[Link]](http://proceedings.mlr.press/v80/alaa18a.html) | ICML 2018 | [alg/causal_multitask_gaussian_processes_ite](alg/causal_multitask_gaussian_processes_ite)
GAIN: Missing Data Imputation using Generative Adversarial Nets [[Link]](http://proceedings.mlr.press/v80/yoon18a.html) | ICML 2018 | [alg/gain/](alg/gain/)
RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks [[Link]](http://proceedings.mlr.press/v80/yoon18b.html) | ICML 2018 | [alg/RadialGAN](alg/RadialGAN)
GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets [[Link]](https://openreview.net/forum?id=ByKWUeWA-) | ICLR 2018 | [alg/ganite](alg/ganite)
DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks [[Link]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16160/15945) | AAAI 2018 | [alg/deephit](alg/deephit)
DAGs with NO TEARS: Continuous Optimization for Structure Learning [[Link]](https://arxiv.org/abs/1803.01422) | NeurIPS 2018 | [alg/castle](alg/castle)
INVASE: Instance-wise Variable Selection using Neural Networks [[Link]](https://openreview.net/forum?id=BJg_roAcK7) | ICLR 2019 | [alg/invase](alg/invase)
PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees [[Link]](https://openreview.net/forum?id=S1zk9iRqF7) | ICLR 2019 | [alg/pategan](alg/pategan)
KnockoffGAN: Generating Knockoffs for Feature Selection using Generative Adversarial Networks [[Link]](https://openreview.net/forum?id=ByeZ5jC5YQ) | ICLR 2019 | [alg/knockoffgan](alg/knockoffgan)
ASAC: Active Sensing using Actor-Critic Models [[Link]](https://arxiv.org/abs/1906.06796) | MLHC 2019 | [alg/asac](alg/asac)
Demystifying Black-box Models with Symbolic Metamodels [[Link]](https://papers.nips.cc/paper/2019/hash/567b8f5f423af15818a068235807edc0-Abstract.html) | NeurIPS 2019 | [alg/symbolic_metamodeling](alg/symbolic_metamodeling)
Differentially Private Bagging: Improved Utility and Cheaper Privacy than Subsample-and-Aggregate [[Link]](https://papers.nips.cc/paper/2019/hash/5dec707028b05bcbd3a1db5640f842c5-Abstract.html) | NeurIPS 2019 | [alg/dpbag](alg/dpbag)
Time-series Generative Adversarial Networks [[Link]](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) | NeurIPS 2019 | [alg/timegan](alg/timegan)
Attentive State-Space Modeling of Disease Progression [[Link]](https://papers.nips.cc/paper/2019/hash/1d0932d7f57ce74d9d9931a2c6db8a06-Abstract.html) | NeurIPS 2019 | [alg/attentivess](alg/attentivess)
Conditional Independence Testing using Generative Adversarial Networks [[Link]](https://arxiv.org/abs/1907.04068) | NeurIPS 2019 | [alg/gcit](alg/gcit)
Temporal Quilting for Survival Analysis [[Link]](http://proceedings.mlr.press/v89/lee19a.html) | AISTATS 2019 | [alg/survivalquilts](alg/survivalquilts)
Estimating Counterfactual Treatment Outcomes over Time through Adversarially Balanced Representations [[Link]](https://openreview.net/forum?id=BJg866NFvB) | ICLR 2020 | [alg/counterfactual_recurrent_network](alg/counterfactual_recurrent_network)
Learning Sparse Nonparametric DAGs [[Link]](http://proceedings.mlr.press/v108/zheng20a.html) | AISTATS 2020 | [alg/castle](alg/castle)
Contextual Constrained Learning for Dose-Finding Clinical Trials [[Link]](https://arxiv.org/abs/2001.02463) | AISTATS 2020 | [alg/c3t_budgets](alg/c3t_budgets)
Learning Overlapping Representations for the Estimation of Individualized Treatment Effects [[Link]](https://arxiv.org/abs/2001.04754) | AISTATS 2020 | [alg/dklite](alg/dklite)
Learning Dynamic and Personalized Comorbidity Networks from Event Data using Deep Diffusion Processes [[Link]](https://arxiv.org/abs/2001.02585) | AISTATS 2020 | [alg/dynamic_disease_network_ddp](alg/dynamic_disease_network_ddp)
Stepwise Model Selection for Sequence Prediction via Deep Kernel Learning [[Link]](https://arxiv.org/abs/2001.03898) | AISTATS 2020 | [alg/smsdkl](alg/smsdkl)
Temporal Phenotyping using Deep Predicting Clustering of Disease Progression [[Link]](http://proceedings.mlr.press/v119/lee20h.html) | ICML 2020 | [alg/ac_tpc](alg/ac_tpc)
Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders [[Link]](http://proceedings.mlr.press/v119/bica20a.html) | ICML 2020 | [alg/time_series_deconfounder](alg/time_series_deconfounder)
Discriminative Jackknife: Quantifying Uncertainty in Deep Learning via Higher-Order Influence Functions [[Link]](http://proceedings.mlr.press/v119/alaa20a.html) | ICML 2020 | [alg/discriminative-jackknife](alg/discriminative-jackknife)
Frequentist Uncertainty in Recurrent Neural Networks via Blockwise Influence Functions [[Link]](http://proceedings.mlr.press/v119/alaa20b.html) | ICML 2020 | [alg/rnn-blockwise-jackknife](alg/rnn-blockwise-jackknife)
Unlabelled Data Improves Bayesian Uncertainty Calibration under Covariate Shift [[Link]](http://proceedings.mlr.press/v119/chan20a.html) | ICML 2020 | [alg/transductive_dropout](alg/transductive_dropout)
Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN) [[Link]](https://ieeexplore.ieee.org/document/9034117) | IEEE | [alg/adsgan](alg/adsgan)
When and How to Lift the Lockdown? Global COVID-19 Scenario Analysis and Policy Assessment using Compartmental Gaussian Processes [[Link]](https://vanderschaar-lab.com/papers/NeurIPS2020_CGP.pdf) | NeurIPS 2020 | [alg/compartmental_gp](alg/compartmental_gp)
Strictly Batch Imitation Learning by Energy-based Distribution Matching [[Link]](https://arxiv.org/abs/2006.14154) | NeurIPS 2020 | [alg/edm](alg/edm)
Gradient Regularized V-Learning for Dynamic Treatment Regimes [[Link]](https://vanderschaar-lab.com/papers/NeurIPS2020_GRV.pdf) | NeurIPS 2020 | [alg/grv](alg/grv)
OrganITE: Optimal transplant donor organ offering using an individual treatment effect [[Link]](https://vanderschaar-lab.com/papers/NeurIPS2020_OrganITE.pdf) | NeurIPS 2020 | [alg/organite](alg/organite)
Robust Recursive Partitioning for Heterogeneous Treatment Effects with Uncertainty Quantification [[Link]](https://arxiv.org/abs/2006.07917) | NeurIPS 2020 | [alg/r2p-hte](alg/r2p-hte)
Estimating the Effects of Continuous-valued Interventions using Generative Adversarial Networks [[Link]](https://arxiv.org/abs/2002.12326) | NeurIPS 2020 | [alg/scigan](alg/scigan)
Learning outside the Black-Box: The pursuit of interpretable models [[Link]](https://arxiv.org/abs/2011.08596) | NeurIPS 2020 | [alg/Symbolic-Pursuit](alg/Symbolic-Pursuit)
VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain [[Link]](https://papers.nips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) | NeurIPS 2020 | [alg/vime](alg/vime)
Scalable Bayesian Inverse Reinforcement Learning [[Link]](https://openreview.net/pdf?id=4qR3coiNaIv) | ICLR 2021 | [alg/scalable-birl](alg/scalable-birl)
Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms[[Link]](https://arxiv.org/abs/2101.10943)| AISTATS 2021|[alg/CATENets](alg/CATENets)
<br/>

* Details of apps and other software is listed below:

App/Software [[Link]](#) | Description | Publication | Code
--- | --- | --- | --- 
Adjutorium COVID-19 [[Link]](https://www.vanderschaar-lab.com/paper-on-covid-19-hospital-capacity-planning-published-in-machine-learning/) | Adjutorium COVID-19: an AI-powered tool that accurately predicts how COVID-19 will impact resource needs (ventilators, ICU beds, etc.) at the individual patient level and the hospital level | - | [app/adjutorium-covid19-public](app/adjutorium-covid19-public)
Clairvoyance [[Link]](https://www.vanderschaar-lab.com/clairvoyance-alpha-the-first-unified-end-to-end-automl-pipeline-for-time-series-data/) | Clairvoyance: A Pipeline Toolkit for Medical Time Series | - | [clairvoyance repository](https://github.com/vanderschaarlab/clairvoyance)
Hide-and-Seek Privacy Challenge [[Link]](http://www.vanderschaar-lab.com/privacy-challenge/) | Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data | [NeurIPS 2020 competition track](https://arxiv.org/abs/2007.12087) | [app/hide-and-seek](app/hide-and-seek)

## Citations
Please cite the *the applicable papers* and [van der Schaar Lab repository](https://github.com/vanderschaarlab/mlforhealthlabpub/) if you use the software.

## License
Copyright **2019-2021** van der Schaar Lab.

This software is released under the [3-Clause BSD license](https://opensource.org/licenses/BSD-3-Clause) unless mentioned otherwise by the respective algorithms and apps.

## Installation instructions
*See individual algorithm and app directories for installation instructions.*

See also [doc/install.md](doc/install.md) for common installation instructions.

## Tutorials and or examples
*See individual algorithm and app directories for tutorials and examples.*

## Data
Data files (as well as other large files such as saved models etc.) can be downloaded as per instructions in the `DATA-*.md` (see e.g. [DATA-PUBLIC.md](./DATA-PUBLIC.md)) files found in the corresponding directories.

## More info
For more information on the van der Schaar Lab’s work, visit [our homepage](https://www.vanderschaar-lab.com/).

## References
*See individual algorithm and app directories for references.*
