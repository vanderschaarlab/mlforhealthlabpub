import numpy as np
from datasets import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def data_loader(dataset_name, random_seed=42, fix=False, test_ratio=0.2):
    if dataset_name in ["bostonHousing", "energy", "wine-quality-red", "yacht", "meps_19", "meps_20", "meps_21", "star",
                        "facebook_1", "facebook_2", "bio", 'blog_data', "concrete", "bike", "community"]:


        X, y = datasets.GetDataset(dataset_name, base_path='datasets/')

        # Dataset is divided into test and train data based on test_ratio parameter
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        # Input dimensions
        n_train = X_train.shape[0]
        in_shape = X_train.shape[1]
        idx = np.array(range(n_train))

        # Features are normalized to (0,1)
        scalerX = MinMaxScaler(feature_range=(0, 1))
        scalerX = scalerX.fit(X_train[idx])
        X_train = scalerX.transform(X_train)
        X_test = scalerX.transform(X_test)

        # Scale the labels by dividing each by the mean absolute response
        mean_ytrain = np.mean(np.abs(y_train[idx]))
        y_train = np.squeeze(y_train) / mean_ytrain
        y_test = np.squeeze(y_test) / mean_ytrain

        return X_train, y_train, X_test, y_test

    else:
        raise AssertionError('Error: wrong data name')


def mixup(X_train, random_seed=42):
    size_mix = len(X_train)
    np.random.seed(random_seed)
    id1 = np.random.randint(0, len(X_train), size=size_mix)
    np.random.seed(random_seed)
    id2 = np.random.randint(0, len(X_train), size=size_mix)
    np.random.seed(random_seed)
    lam = np.random.beta(0.2, 0.2, size=(size_mix, 1))
    X_mix = lam*X_train[id1] + (1-lam)*X_train[id2]
    return X_mix
