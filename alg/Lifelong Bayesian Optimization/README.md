# LBO
Required library: Tensorflow, xgboost, matplotlib, pandas

LBO.py: The model of Lifelong Bayesian Optimization(LBO)

LBO_test1.py: Branin functions optimization with LBO

LBO_test2.py: Hyperparameter optimization with LBO on the UNOS2 dataset

Evaluate.py: Black-box functions evaluation (sample a point from Branin function or try a hyperparameter on UNOS)

In LBO_test1.py, we minimize two Branin functions. LBO should minimize the second function faster than the first function. When the program finishes running, it will generate a plot comparing the optimization trajectory of the two Branin functions, denoted by 'f_0: No Transfer' and 'f_1: Transfer'. We expect to see 'f_1: Transfer' will converge faster than 'f_0: No Transfer'.

LBO_test2.py is similar to LBO_test1.py. We optimize the AUC on the UNOS dataset twice. We expect to see the AUC of 'f_1: Transfer' will converge faster than 'f_0: No Transfer'.

If you want to try LBO on other datasets, replace the data "X_train" and "Y_train" in LBO_test2.py by another dataset.
