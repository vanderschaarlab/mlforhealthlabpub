import argparse
import csv
import pickle
from symbolic_pursuit.models import SymbolicRegressor
from datasets.data_loader_UCI import data_loader, mixup
from experiments.train_model import train_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from time import strftime, gmtime


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='wine-quality-red', type=str
    )
    parser.add_argument(
        '--test_ratio', default=0.2, type=float, help='percentage of test examples')
    parser.add_argument(
        '--model', default=None, help='a black box model to interpret'
    )
    parser.add_argument(
        '--model_type', default='MLP', type=str, help='type of black-box (MLP, KNN, ...)'
    )
    parser.add_argument(
        '--verbosity', default=True, type=bool, help='true if the optimization process should be detailed')
    parser.add_argument(
        '--loss_tol', default=1.0e-3, type=float, help='the tolerance for the loss under which the pursuit stops'
    )
    parser.add_argument(
        '--ratio_tol', default=0.9, type=float, help='a new term is added only if new_loss / old_loss < ratio_tol'
    )
    parser.add_argument(
        '--maxiter', default=100, type=int, help='maximum number of iterations for optimization'
    )
    parser.add_argument(
        '--eps', default=1.0e-5, type=float, help='small number used for numerical stability'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42, help='random seed for reproducibility'
    )
    return parser.parse_args()


if __name__ == '__main__':

    # Extract arguments from the parser

    args = init_arg()
    dataset_name = args.dataset
    model = args.model
    model_type = args.model_type
    test_ratio = args.test_ratio
    verbosity = args.verbosity
    loss_tol = args.loss_tol
    ratio_tol = args.ratio_tol
    maxiter = args.maxiter
    eps = args.eps
    random_seed = args.random_seed
    print("\nWelcome to this experiment evaluating the performance of symbolic modeling. \n"
          + "This experiment uses the black-box {} on the dataset {}. \n".format(model_type, dataset_name)
          + "The ratio of test examples is test_ratio={}. \n".format(test_ratio))

    # Train the model (if no model is given) and the symbolic model

    X_train, y_train, X_test, y_test = data_loader(dataset_name, random_seed=random_seed, test_ratio=test_ratio)
    X_mixup = mixup(X_train, random_seed=random_seed)
    if model is None:
        model = train_model(X_train, y_train, black_box=model_type, random_seed=random_seed)
    else:
        model_type = model.__class__.__name__
    symbolic_model = SymbolicRegressor(verbosity=verbosity, loss_tol=loss_tol, ratio_tol=ratio_tol,
                                       maxiter=maxiter, eps=eps, random_seed=random_seed)
    symbolic_model.fit(model.predict, X_mixup)

    # Compute the metrics

    model_mse = mean_squared_error(y_test, model.predict(X_test))
    symbolic_mse = mean_squared_error(y_test, symbolic_model.predict(X_test))
    model_symbolic_mse = mean_squared_error(model.predict(X_test), symbolic_model.predict(X_test))
    model_r2 = r2_score(y_test, model.predict(X_test))
    symbolic_r2 = r2_score(y_test, symbolic_model.predict(X_test))
    model_symbolic_r2 = r2_score(model.predict(X_test), symbolic_model.predict(X_test))
    symbolic_nterms = len(symbolic_model.terms_list)

    # Save everything

    time_str = strftime('%Y-%m-%d %H:%M:%S', gmtime())
    with open('experiments/dataset_results.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        csv_writer.writerow([time_str, dataset_name, model_type,
                             model_mse, symbolic_mse, model_symbolic_mse,
                             model_r2, symbolic_r2, model_symbolic_r2, symbolic_nterms])


    with open("experiments/models/{}_{}_{}.pickle".format(dataset_name, model_type, time_str), 'wb') as filename:
        save_tuple = (model, symbolic_model)
        pickle.dump(save_tuple, filename)
