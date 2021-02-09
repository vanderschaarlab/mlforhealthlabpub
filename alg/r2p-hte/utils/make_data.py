import numpy as np
import pandas as pd


def sample_SynthA(treat_prob=0.5, sample_no=1000):
    """
        Generate synthetic data from Athey and Imbens
    """
    noise1 = np.random.normal(0, 0.01, sample_no)
    noise0 = np.random.normal(0, 0.01, sample_no)

    w = np.random.uniform(0, 1, sample_no)
    w = w <= treat_prob
    y = np.zeros(sample_no)

    model_treatment = lambda eta, x, kappa, noise: eta(x) + 1 / 2 * (kappa(x)) + noise
    model_control = lambda eta, x, kappa, noise: eta(x) - 1 / 2 * (kappa(x)) + noise

    x = np.random.normal(0, 1, [sample_no, 2])
    eta = lambda x: 1 / 2 * x[:, 0] + x[:, 1]
    kappa = lambda x: 1 / 2 * x[:, 0]
    y1 = model_treatment(eta, x, kappa, noise1)
    y0 = model_control(eta, x, kappa, noise0)
    y1 = y1.squeeze()
    y0 = y0.squeeze()
    y[w == 1] = y1[w == 1]
    y[w == 0] = y0[w == 0]
    tau = y1 - y0

    return x, w, y, y1, y0, tau


def sample_SynthB(train_sample_no=300, test_sample_no=1000):
    sample_no = train_sample_no + test_sample_no
    X = np.round(np.random.normal(size=(sample_no, 1), loc=66.0, scale=4.1))  # age
    X = np.block([X, np.round(
        np.random.normal(size=(sample_no, 1), loc=6.2, scale=1.0) * 10.0) / 10.0])  # white blood cell count
    X = np.block(
        [X, np.round(np.random.normal(size=(sample_no, 1), loc=0.8, scale=0.1) * 10.0) / 10.0])  # Lymphocyte count
    X = np.block([X, np.round(np.random.normal(size=(sample_no, 1), loc=183.0, scale=20.4))])  # Platelet count
    X = np.block([X, np.round(np.random.normal(size=(sample_no, 1), loc=68.0, scale=6.6))])  # Serum creatinine
    X = np.block(
        [X, np.round(np.random.normal(size=(sample_no, 1), loc=31.0, scale=5.1))])  # Aspartete aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(sample_no, 1), loc=26.0, scale=5.1))])  # Alanine aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(sample_no, 1), loc=339.0, scale=51))])  # Lactate dehydrogenase
    X = np.block([X, np.round(np.random.normal(size=(sample_no, 1), loc=76.0, scale=21))])  # Creatine kinase
    X = np.block([X, np.floor(np.random.uniform(size=(sample_no, 1)) * 11) + 4])  # Time from study 4~14
    TIME = X[:, 9]

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    W = np.random.binomial(1, 0.5, size=sample_no)

    # sample random coefficients
    coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
    BetaB = np.random.choice(coeffs_, size=9, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])

    MU_0 = np.dot(X[:, 0:9], BetaB)
    MU_1 = np.dot(X[:, 0:9], BetaB)

    logi0 = lambda x: 1 / (1 + np.exp(-(x - 9))) + 5
    logi1 = lambda x: 5 / (1 + np.exp(-(x - 9)))

    MU_0 = MU_0 + logi0(TIME)
    MU_1 = MU_1 + logi1(TIME)

    Y_0 = np.random.normal(scale=0.1, size=len(X)) + MU_0
    Y_1 = np.random.normal(scale=0.1, size=len(X)) + MU_1

    train_index = list(np.random.choice(range(sample_no), train_sample_no, replace=False))
    test_index = list(set(list(range(sample_no))) - set(train_index))

    X_train = X[train_index]
    W_train = W[train_index]

    Y_0_train = Y_0[train_index]
    Y_1_train = Y_1[train_index]

    Y_train = W_train * Y_1_train + (1 - W_train) * Y_0_train
    T_true_train = Y_1[train_index] - Y_0[train_index]
    Y_cf_train = W_train * Y_0_train + (1 - W_train) * Y_1_train

    X_test = X[test_index]
    W_test = W[test_index]

    Y_0_test = Y_0[test_index]
    Y_1_test = Y_1[test_index]

    Y_test = W_test * Y_1_test + (1 - W_test) * Y_0_test
    Y_cf_test = W_test * Y_0_test + (1 - W_test) * Y_1_test

    T_true_test = Y_1_test - Y_0_test

    train_data = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    test_data = (X_test, W_test, Y_test, Y_0_test, Y_1_test, Y_cf_test, T_true_test)

    return train_data, test_data


def sample_IHDP(fn_data, test_frac=0.2, noise=0.1):
    Dataset = pd.read_csv(fn_data, header=None)
    col = ["Treatment", "Response", "Y_CF", "mu0", "mu1", ]

    for i in range(1, 26):
        col.append("X" + str(i))
    Dataset.columns = col
    Dataset.head()

    num_samples = len(Dataset)
    train_size = int(np.floor(num_samples * (1 - test_frac)))

    train_index = list(np.random.choice(range(num_samples), train_size, replace=False))
    test_index = list(set(list(range(num_samples))) - set(train_index))

    feat_name = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'

    Data_train = Dataset.loc[Dataset.index[train_index]]
    Data_test = Dataset.loc[Dataset.index[test_index]]

    X_train = np.array(Data_train[feat_name.split()])
    W_train = np.array(Data_train['Treatment'])

    Y_0_train = np.array(np.random.normal(scale=noise, size=len(X_train)) + Data_train['mu0'])
    Y_1_train = np.array(np.random.normal(scale=noise, size=len(X_train)) + Data_train['mu1'])

    Y_train = W_train * Y_1_train + (1 - W_train) * Y_0_train
    Y_cf_train = W_train * Y_0_train + (1 - W_train) * Y_1_train

    T_true_train = np.array(Data_train['mu1'] - Data_train['mu0'])

    X_test = np.array(Data_test[feat_name.split()])
    W_test = np.array(Data_test['Treatment'])

    Y_0_test = np.array(np.random.normal(scale=noise, size=len(X_test)) + Data_test['mu0'])
    Y_1_test = np.array(np.random.normal(scale=noise, size=len(X_test)) + Data_test['mu1'])
    Y_test = W_test * Y_1_test + (1 - W_test) * Y_0_test
    Y_cf_test = W_test * Y_0_test + (1 - W_test) * Y_1_test

    T_true_test = Y_1_test - Y_0_test

    train_data = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    test_data = (X_test, W_test, Y_test, Y_0_test, Y_1_test, Y_cf_test, T_true_test)

    return train_data, test_data


def sample_CPP(file_path, train_sample_no=500, test_sample_no=1000):
    dataset = pd.DataFrame(pd.read_csv(file_path, sep=",", index_col=0))
    dataset.x_2 = [ord(x) - 64 for x in dataset.x_2]
    dataset.x_21 = [ord(x) - 64 for x in dataset.x_21]
    dataset.x_24 = [ord(x) - 64 for x in dataset.x_24]

    q_low = dataset["y0"].quantile(0.01)
    q_hi = dataset["y0"].quantile(0.99)
    dataset = dataset.loc[(dataset["y0"] < q_hi) & (dataset["y0"] > q_low)]
    q_low = dataset["y1"].quantile(0.01)
    q_hi = dataset["y1"].quantile(0.99)
    dataset = dataset.loc[(dataset["y1"] < q_hi) & (dataset["y1"] > q_low)]

    treat_no = int(np.floor((train_sample_no + test_sample_no) * 0.35))
    control_no = (train_sample_no + test_sample_no) - treat_no
    data_sample = dataset[dataset['z'] == 1].sample(treat_no)
    data_sample = pd.concat(
        [data_sample, dataset[dataset['z'] == 0].sample(control_no)])

    data_sample.iloc[:, :-7] = normalize(data_sample.iloc[:, :-7])
    data_sample.iloc[:, -5:-1] = data_sample.iloc[:, -5:-1] / 10
    data_sample['y'] = data_sample['y'] / 10
    train_index = list(np.random.choice(range(train_sample_no + test_sample_no), train_sample_no, replace=False))
    test_index = list(set(list(range(train_sample_no + test_sample_no))) - set(train_index))

    train_data = data_sample.loc[data_sample.index[train_index]]
    test_data = data_sample.loc[data_sample.index[test_index]]

    X_train = np.array(train_data.iloc[:, :-7])
    W_train = np.array(train_data['z'])
    Y_train = np.array(train_data['y'])
    Y_cf_train = np.array(train_data['z'] * train_data['y1'] + (1 - train_data['z']) * train_data['y0'])

    Y_0_train = np.array(train_data['y0'])
    Y_1_train = np.array(train_data['y1'])
    T_true_train = Y_1_train - Y_0_train

    X_test = np.array(test_data.iloc[:, :-7])
    W_test = np.array(test_data['z'])
    Y_test = np.array(test_data['z'] * test_data['y1'] + (1 - test_data['z']) * test_data['y0'])
    Y_cf_test = np.array(test_data['z'] * test_data['y1'] + (1 - test_data['z']) * test_data['y0'])

    Y_0_test = np.array(test_data['y0'])
    Y_1_test = np.array(test_data['y1'])

    T_true_test = Y_1_test - Y_0_test

    train_data = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    test_data = (X_test, W_test, Y_test, Y_0_test, Y_1_test, Y_cf_test, T_true_test)

    return train_data, test_data


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def normalize_mean(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (result[feature_name] - result[feature_name].mean()) / result[feature_name].std()
    return result
