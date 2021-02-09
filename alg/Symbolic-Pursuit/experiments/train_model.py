from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def train_model(X_train, y_train, black_box="MLP", random_seed=42):
    # Fits a given black-box model to the training set
    if black_box == "MLP":
        model = MLPRegressor(random_state=random_seed)
    elif black_box == "KNN":
        model = KNeighborsRegressor(random_state=random_seed)
    elif black_box == "SVM":
        model = SVR(random_state=random_seed)
    elif black_box == "XGB":
        model = XGBRegressor(objective='reg:squarederror', random_state=random_seed)
    elif black_box == "Tree":
        model = DecisionTreeRegressor(random_state=random_seed)
    elif black_box == "RF":
        model = RandomForestRegressor(random_state=random_seed)
    else:
        raise NameError("black-box model type unknown")
    model.fit(X_train, y_train)
    return model
