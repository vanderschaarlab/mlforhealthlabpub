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
The publications and the corresponding locations in the repo are listed below:

Paper [[Link]](#) | Journal/Conference | Code
--- | --- | ---
Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes [[Link]](https://proceedings.neurips.cc/paper/2017/hash/6a508a60aa3bf9510ea6acb021c94b48-Abstract.html) | NIPS 2017 | [alg/causal_multitask_gaussian_processes_ite](alg/causal_multitask_gaussian_processes_ite)
Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks [[Link]](https://proceedings.neurips.cc/paper/2017/hash/861dc9bd7f4e7dd3cccd534d0ae2a2e9-Abstract.html) | NIPS 2017 | [alg/dgp_survival](alg/dgp_survival)
AutoPrognosis: Automated Clinical Prognostic Modeling via Bayesian Optimization with Structured Kernel Learning [[Link]](https://icml.cc/Conferences/2018/Schedule?showEvent=2050) | ICML 2018 | [alg/autoprognosis](alg/autoprognosis)
Limits of Estimating Heterogeneous Treatment Effects: Guidelines for Practical Algorithm Design [[Link]](http://proceedings.mlr.press/v80/alaa18a.html) | ICML 2018 | [alg/causal_multitask_gaussian_processes_ite](alg/causal_multitask_gaussian_processes_ite)
GAIN: Missing Data Imputation using Generative Adversarial Nets [[Link]](http://proceedings.mlr.press/v80/yoon18a.html) | ICML 2018 | [alg/gain](alg/gain)
RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks [[Link]](http://proceedings.mlr.press/v80/yoon18b.html) | ICML 2018 | [alg/RadialGAN](alg/RadialGAN)
GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets [[Link]](https://openreview.net/forum?id=ByKWUeWA-) | ICLR 2018 | [alg/ganite](alg/ganite)
Deep Sensing: Active Sensing using Multi-directional Recurrent Neural Networks [[Link]](https://openreview.net/forum?id=r1SnX5xCb) | ICLR 2018 | [alg/DeepSensing (MRNN)](alg/DeepSensing%20(MRNN))
DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks [[Link]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16160/15945) | AAAI 2018 | [alg/deephit](alg/deephit)
INVASE: Instance-wise Variable Selection using Neural Networks [[Link]](https://openreview.net/forum?id=BJg_roAcK7) | ICLR 2019 | [alg/invase](alg/invase)
PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees [[Link]](https://openreview.net/forum?id=S1zk9iRqF7) | ICLR 2019 | [alg/pategan](alg/pategan)
KnockoffGAN: Generating Knockoffs for Feature Selection using Generative Adversarial Networks [[Link]](https://openreview.net/forum?id=ByeZ5jC5YQ) | ICLR 2019 | [alg/knockoffgan](alg/knockoffgan)
ASAC: Active Sensing using Actor-Critic Models [[Link]](https://arxiv.org/abs/1906.06796) | MLHC 2019 | [alg/asac](alg/asac)
Demystifying Black-box Models with Symbolic Metamodels [[Link]](https://papers.nips.cc/paper/2019/hash/567b8f5f423af15818a068235807edc0-Abstract.html) | NeurIPS 2019 | [alg/symbolic_metamodeling](alg/symbolic_metamodeling)
Differentially Private Bagging: Improved Utility and Cheaper Privacy than Subsample-and-Aggregate [[Link]](https://papers.nips.cc/paper/2019/hash/5dec707028b05bcbd3a1db5640f842c5-Abstract.html) | NeurIPS 2019 | [alg/dpbag](alg/dpbag)
Time-series Generative Adversarial Networks [[Link]](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) | NeurIPS 2019 | [alg/timegan](alg/timegan)
Attentive State-Space Modeling of Disease Progression [[Link]](https://papers.nips.cc/paper/2019/hash/1d0932d7f57ce74d9d9931a2c6db8a06-Abstract.html) | NeurIPS 2019 | [alg/attentivess](alg/attentivess)
Conditional Independence Testing using Generative Adversarial Networks [[Link]](https://arxiv.org/abs/1907.04068) | NeurIPS 2019 | [alg/gcit](alg/gcit)
Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis with Competing Risks based on Longitudinal Data [[Link]](https://ieeexplore.ieee.org/document/8681104) | IEEE | [alg/dynamic_deephit](alg/dynamic_deephit)
Temporal Quilting for Survival Analysis [[Link]](http://proceedings.mlr.press/v89/lee19a.html) | AISTATS 2019 | [alg/survivalquilts](alg/survivalquilts)
Estimating Counterfactual Treatment Outcomes over Time through Adversarially Balanced Representations [[Link]](https://openreview.net/forum?id=BJg866NFvB) | ICLR 2020 | [alg/counterfactual_recurrent_network](alg/counterfactual_recurrent_network)
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
CASTLE: Regularization via Auxiliary Causal Graph Discovery [[Link]](https://arxiv.org/abs/2009.13180) | NeurIPS 2020 | [alg/castle](alg/castle)
OrganITE: Optimal transplant donor organ offering using an individual treatment effect [[Link]](https://vanderschaar-lab.com/papers/NeurIPS2020_OrganITE.pdf) | NeurIPS 2020 | [alg/organite](alg/organite)
Robust Recursive Partitioning for Heterogeneous Treatment Effects with Uncertainty Quantification [[Link]](https://arxiv.org/abs/2006.07917) | NeurIPS 2020 | [alg/r2p-hte](alg/r2p-hte)
Estimating the Effects of Continuous-valued Interventions using Generative Adversarial Networks [[Link]](https://arxiv.org/abs/2002.12326) | NeurIPS 2020 | [alg/scigan](alg/scigan)
Learning outside the Black-Box: The pursuit of interpretable models [[Link]](https://arxiv.org/abs/2011.08596) | NeurIPS 2020 | [alg/Symbolic-Pursuit](alg/Symbolic-Pursuit)
VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain [[Link]](https://papers.nips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) | NeurIPS 2020 | [alg/vime](alg/vime)
Scalable Bayesian Inverse Reinforcement Learning [[Link]](https://openreview.net/pdf?id=4qR3coiNaIv) | ICLR 2021 | [alg/scalable-birl](alg/scalable-birl)
Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms [[Link]](https://arxiv.org/abs/2101.10943) | AISTATS 2021 | [alg/CATENets](https://github.com/vanderschaarlab/CATENets)
Learning Matching Representations for Individualized Organ Transplantation Allocation [[Link]](https://arxiv.org/abs/2101.11769) | AISTATS 2021| [alg/MatchingRep](alg/MatchingRep)
Explaining by Imitating: Understanding Decisions by Interpretable Policy Learning [[Link]](https://openreview.net/forum?id=unI5ucw_Jk) | ICLR 2021 | [alg/interpole](alg/interpole)
Inverse Decision Modeling: Learning Interpretable Representations of Behavior [[Link]](http://proceedings.mlr.press/v139/jarrett21a.html) | ICML 2021 | [alg/ibrc](alg/ibrc)
Policy Analysis using Synthetic Controls in Continuous-Time [[Link]](http://proceedings.mlr.press/v139/bellot21a/bellot21a.pdf) | ICML 2021 | [alg/Synthetic-Controls-in-Continuous-Time](https://github.com/vanderschaarlab/Synthetic-Controls-in-Continuous-Time/)
Learning Queueing Policies for Organ Transplantation Allocation using Interpretable Counterfactual Survival Analysis [[Link]](http://proceedings.mlr.press/v139/berrevoets21a/berrevoets21a.pdf) | ICML 2021 | [alg/organsync](https://github.com/vanderschaarlab/organsync/)
Explaining Time Series Predictions with Dynamic Masks [[Link]](http://proceedings.mlr.press/v139/crabbe21a.html) | ICML 2021 | [alg/Dynamask](https://github.com/vanderschaarlab/Dynamask/)
Generative Time-series Modeling with Fourier Flows [[Link]](https://openreview.net/forum?id=PpshD0AXfA) | ICLR 2021 | [alg/Fourier-flows](https://github.com/vanderschaarlab/Fourier-flows/)
On Inductive Biases for Heterogeneous Treatment Effect Estimation [[Link]](https://arxiv.org/pdf/2106.03765.pdf) | NeurIPS 2021 | [alg/CATENets](https://github.com/vanderschaarlab/CATENets/)
Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation [[Link]](https://openreview.net/pdf?id=FQLzQqGEAH) | NeurIPS 2021 | [alg/CATENets](https://github.com/vanderschaarlab/CATENets/)
The Medkit-Learn(ing) Environment: Medical Decision Modelling through Simulation [[Link]](https://arxiv.org/abs/2106.04240) | NeurIPS 2021 | [alg/medkit-learn](https://github.com/vanderschaarlab/medkit-learn)
MIRACLE: Causally-Aware Imputation via Learning Missing Data Mechanisms [[Link]](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=27670) | NeurIPS 2021 | [alg/MIRACLE](https://github.com/vanderschaarlab/MIRACLE)
DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks [[Link]](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=27552) | NeurIPS 2021 | [alg/DECAF](https://github.com/vanderschaarlab/DECAF)
Explaining Latent Representations with a Corpus of Examples [[Link]](https://arxiv.org/abs/2110.15355) | NeurIPS 2021 | [alg/Simplex](https://github.com/vanderschaarlab/Simplex)
Closing the loop in medical decision support by understanding clinical decision-making: A case study on organ transplantation [[Link]](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=26815) | NeurIPS 2021 | [alg/iTransplant](https://github.com/vanderschaarlab/iTransplant)
Integrating Expert ODEs into Neural ODEs: Pharmacology and Disease Progression [[Link]](https://papers.neurips.cc/paper/2021/hash/5ea1649a31336092c05438df996a3e59-Abstract.html) | NeurIPS 2021 | [alg/Hybrid-ODE-NeurIPS-2021](https://github.com/vanderschaarlab/Hybrid-ODE-NeurIPS-2021)
SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes [[Link]](https://proceedings.neurips.cc/paper/2021/hash/19485224d128528da1602ca47383f078-Abstract.html) | NeurIPS 2021 | [alg/SyncTwin-NeurIPS-2021](https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021)
Conformal Time-series Forecasting [[Link]](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html) | NeurIPS 2021 | [alg/conformal-rnn](https://github.com/vanderschaarlab/conformal-rnn/tree/master)
Estimating Multi-cause Treatment Effects via Single-cause Perturbation [[Link]](https://proceedings.neurips.cc/paper/2021/hash/c793b3be8f18731f2a4c627fb3c6c63d-Abstract.html) | NeurIPS 2021 | [alg/Single-Cause-Perturbation-NeurIPS-2021](https://github.com/vanderschaarlab/Single-Cause-Perturbation-NeurIPS-2021/)
Invariant Causal Imitation Learning for Generalizable Policies [[Link]](https://papers.nips.cc/paper/2021/file/204904e461002b28511d5880e1c36a0f-Paper.pdf) | NeurIPS 2021 | [alg/Invariant-Causal-Imitation-Learning](https://github.com/vanderschaarlab/Invariant-Causal-Imitation-Learning/tree/main)
Inferring Lexicographically-Ordered Rewards from Preferences [[Link]](https://ojs.aaai.org/index.php/AAAI/article/view/20516) | AAAI 2022 | [alg/lori](https://github.com/vanderschaarlab/lori)
Inverse Online Learning: Understanding Non-Stationary and Reactionary Policies [[Link]](https://openreview.net/forum?id=DYypjaRdph2) | ICLR 2022 | [alg/inverse-online](https://github.com/vanderschaarlab/inverse-online)
D-CODE: Discovering Closed-form ODEs from Observed Trajectories [[Link]](https://openreview.net/forum?id=wENMvIsxNN) | ICLR 2022 | [alg/D-CODE-ICLR-2022](https://github.com/vanderschaarlab/D-CODE-ICLR-2022)
Neural graphical modelling in continuous-time: consistency guarantees and algorithms [[Link]](https://openreview.net/forum?id=SsHBkfeRF9L) | ICLR 2022 | [alg/Graphical-modelling-continuous-time](https://github.com/vanderschaarlab/Graphical-modelling-continuous-time)
Label-Free Explainability for Unsupervised Models [[Link]](https://proceedings.mlr.press/v162/crabbe22a) | ICML 2022 | [alg/Label-Free-XAI](https://github.com/vanderschaarlab/Label-Free-XAI)
Inverse Contextual Bandits: Learning How Behavior Evolves over Time [[Link]](https://proceedings.mlr.press/v162/huyuk22a.html) | ICML 2022 | [alg/invconban](https://github.com/vanderschaarlab/invconban)
Data-SUITE: Data-centric identification of in-distribution incongruous examples [[Link]](https://proceedings.mlr.press/v162/seedat22a.html) | ICML 2022 | [alg/Data-SUITE](https://github.com/vanderschaarlab/Data-SUITE)
Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations [[Link]](https://proceedings.mlr.press/v162/seedat22b) | ICML 2022 | [alg/TE-CDE](https://github.com/vanderschaarlab/TE-CDE)
Concept Activation Regions: A Generalized Framework For Concept-Based Explanations[[Link]](https://arxiv.org/abs/2209.11222) | NeurIPS 2022 | [alg/CARs](https://github.com/vanderschaarlab/CARs)
Benchmarking Heterogeneous Treatment Effect Models through the Lens of Interpretability[[Link]](https://arxiv.org/abs/2206.08363) | NeurIPS 2022 | [alg/ITErpretability](https://github.com/vanderschaarlab/ITErpretability))
<br/>

Details of apps and other software is listed below:

App/Software [[Link]](#) | Description | Publication | Code
--- | --- | --- | ---
Adjutorium COVID-19 [[Link]](https://www.vanderschaar-lab.com/paper-on-covid-19-hospital-capacity-planning-published-in-machine-learning/) | Adjutorium COVID-19: an AI-powered tool that accurately predicts how COVID-19 will impact resource needs (ventilators, ICU beds, etc.) at the individual patient level and the hospital level | - | [app/adjutorium-covid19-public](app/adjutorium-covid19-public)
Clairvoyance [[Link]](https://www.vanderschaar-lab.com/clairvoyance-alpha-the-first-unified-end-to-end-automl-pipeline-for-time-series-data/) | Clairvoyance: A Pipeline Toolkit for Medical Time Series | - | [clairvoyance repository](https://github.com/vanderschaarlab/clairvoyance)
Hide-and-Seek Privacy Challenge [[Link]](http://www.vanderschaar-lab.com/privacy-challenge/) | Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data | [NeurIPS 2020 competition track](https://arxiv.org/abs/2007.12087) | [app/hide-and-seek](app/hide-and-seek)

## Citations
Please cite the *the applicable papers* and [van der Schaar Lab repository](https://github.com/vanderschaarlab/mlforhealthlabpub/) if you use the software.

## Breakdown by category

**Synthetic data**

* [alg/pategan](alg/pategan)
* [alg/adsgan](alg/adsgan)
* [alg/dpbag](alg/dpbag)
* [alg/timegan](alg/timegan)
* [alg/Fourier-flows](https://github.com/vanderschaarlab/Fourier-flows/)

**More categories to come**

* 🚧

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
