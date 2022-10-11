# ITErpretability


Code Authors: Ioana Bica (ioana.bica@eng.ox.ac.uk),  Alicia Curth (amc253@damtp.cam.ac.uk), Jonathan Crabbé (jc2133@cam.ac.uk)

This repository contains the implementation of ITErpretability, a framework to bemchmark treatment effect neural network estimators with the help of interpretability. For more details, please read our [NeurIPS 2022 paper](https://arxiv.org/abs/2206.08363): 'Benchmarking Heterogeneous Treatment Effect Models through the Lens of Interpretability'.


## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Running experiments 

You can run the experiments using the following commands: 

- Experiment 1: Altering the Strength of Predictive Effects

```bash
python run_experiments.py --experiment_name=predictive_sensitivity
```

- Experiment 2: Incorporating Nonlinearities

```bash
python run_experiments.py --experiment_name=nonlinearity_sensitivity
```

- Experiment 3: The Effect of Confounding

```bash
python run_experiments.py --experiment_name=propensity_sensitivity
```

The results from all experiments are saved in results/. You can then plot the results by running the code in the notebook plot_results.ipynb. 

Note that we use the PyTorch implementations of the different CATE learners from the catenets Python package: https://github.com/AliciaCurth/CATENets.

## 3. Adding datasets 

Our code can be easily extended to add more datasets for evaluation. To add a new dataset, you need to update the 
load() function in src/interpretability/data_loader.py to read the features (X_raw) for the new dataset from the 
dataset file. X_raw needs to have shape [N_d, D_f], where N_d is the number of examples in the dataset and D_f is the 
number of features for each example. 

The experiments have set as default arguments the datasets and the number of important features for each dataset used 
for the results in the paper. To run the experiments with your new dataset, you just need to pass it as a command line 
argument. For instance, to run the predictive_sensitivity with a new dataset named 'new_dataset_name' with N_i predictive 
(for each potential outcome) and prognostic features, you can use: 

```bash
python run_experiments.py --experiment_name='predictive_sensitivity' --dataset_name='new_dataset_name' --num_important_features_list='N_I'
```

## 4. Adding learners 
The code can also be easily used with additional CATE learners. We currently use the PyTorch implementations of the different CATE learners from the 
catenets Python package: https://github.com/AliciaCurth/CATENets. However, if you want to evaluate the ability of new/other CATE learners
to discover predictive features for CATE estimation, you can add it to the learners list in src/interpretability/experiments.py. 
The CATE learner needs to be implemented as a Python Class that inherits the BaseCATEEstimator in the CATENets package: https://github.com/AliciaCurth/CATENets/blob/main/catenets/models/torch/base.py#L514

## 5. Citing

If you use this code, please cite the associated paper:

```
@misc{crabbé2022benchmarking,
      title={Benchmarking Heterogeneous Treatment Effect Models through the Lens of Interpretability}, 
      author={Jonathan Crabbé and Alicia Curth and Ioana Bica and Mihaela van der Schaar},
      year={2022},
      eprint={2206.08363},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
