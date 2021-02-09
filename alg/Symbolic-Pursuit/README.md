# Faithful Pursuit

 Github for the NIPS 2020 paper "Learning outside the black-box: at the pursuit of interpretable models"
 
## Usage

To build a symbolic regressor for a given dataset and a given model (or a given model type),
the following command can be used :

````
 python3 build_interpreter.py [-h] [--dataset DATASET] [--test_ratio TEST_RATIO]
                            [--model MODEL] [--model_type MODEL_TYPE]
                            [--verbosity VERBOSITY] [--loss_tol LOSS_TOL]
                            [--ratio_tol RATIO_TOL] [--maxiter MAXITER]
                            [--eps EPS] [--random_seed RANDOM_SEED]
````

For example, if one would like to train a MLP one the wine-quality-red dataset and then
fit a symbolic regressor with random seed 27, one can use the command

````
python3 build_interpreter --dataset wine-quality-red --model_type MLP --random_seed 27
````
 
 For more details on how to use the module in general, see the 3 enclosed notebooks.
 
 [1. Building a Symbolic Regressor](./1.%20Building%20a%20Symbolic%20Regressor.ipynb)  
 [2. Symbolic Pursuit vs LIME](./2.%20Symbolic%20Pursuit%20vs%20LIME.ipynb)  
 [3. Synthetic experiments with Symbolic Pursuit](./3.%20Synthetic%20experiments%20with%20Symbolic%20Pursuit.ipynb)

## Dependencies

| Name     	| Version  	|
|---	|---	|
| cite2c | 0.2.1 |
| joblib | 0.17.0 |
| jupyter | 1.0.0 |
| lime | 0.2.0.1 |
| numpy | 1.19.1 |
| python | 3.7.9 |
| scikit-learn | 0.23.2 |
| sympy | 1.6.2 |
| xgboost | 1.2.1 |

## References

In our experiments, we used implementations of [LIME](https://github.com/marcotcr/lime), [SHAP](https://github.com/slundberg/shap) and [pysymbolic](https://github.com/ahmedmalaa/Symbolic-Metamodeling)

