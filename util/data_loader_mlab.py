import pandas as pd
import utilmlab
import sys
from sklearn.datasets import load_breast_cancer, fetch_covtype
import logging
import argparse
import os


ds2fn_d = {
    'bc': None,
    'cover': None,
    'breastcancer': None,
    'spam': '{}/spam.csv.gz'.format(utilmlab.get_data_dir()),
    'spambase': '{}/spambase.csv.gz'.format(utilmlab.get_data_dir()),
    'news': '{}/OnlineNewsPopularity.csv.gz'.format(utilmlab.get_data_dir()),
    'newsbin': '{}/OnlineNewsPopularity.csv.gz'.format(
        utilmlab.get_data_dir()),
    'letter': '{}/letter.csv.gz'.format(utilmlab.get_data_dir()),
    'letter-recognition': '{}/letter.csv.gz'.format(utilmlab.get_data_dir()),
    'creditcardfraud': '{}/kaggle_creditcardfraud/creditcard_modified.csv'.format(
        utilmlab.get_data_dir())
}


def is_available(ds):
    if ds in ds2fn_d.keys():
        return True if ds2fn_d[ds] is None else os.path.isfile(ds2fn_d[ds])


def get_available_datasets():
    ds_lst = list()
    for ds in ds2fn_d.keys():
        if is_available(ds):
            ds_lst.append(ds)
    return ds_lst


def get_dataset(dataset, nsample=0):

    fn = None
    labels = []
    features_not_scalable = []
    features_drop = []
    rval = 0
    df = None
    dataset_lst = get_available_datasets()

    if dataset not in get_available_datasets() + ['show']:
        print('error: {} not available: ({})'.format(dataset, dataset_lst))
        sys.exit(1)

    if dataset == 'spam':
        fn = '{}/spam.csv.gz'.format(utilmlab.get_data_dir())
    elif dataset == 'spambase':
        fn = '{}/spambase.csv.gz'.format(utilmlab.get_data_dir())
        labels = ['label']
    elif dataset == 'breastcancer' or dataset == 'bc':
        data = load_breast_cancer()  # get Breast Cancer Dataset
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = 'target'
        df[target] = data.target
        labels = [target]
    elif dataset == 'cover':
        data = fetch_covtype()
        df = pd.DataFrame(data.data)
        target = 'target'
        df[target] = data.target
        labels = [target]
    elif dataset == 'news':
        fn = '{}/OnlineNewsPopularity.csv.gz'.format(
            utilmlab.get_data_dir())
        labels = [' shares']
        features_drop = ['url']
    elif dataset == 'newsbin':
        fn = '{}/OnlineNewsPopularity.csv.gz'.format(
            utilmlab.get_data_dir())
        labels = [' shares']
        features_drop = ['url']
        response_var = labels[0]
        df = pd.read_csv(fn)
        df[response_var] = [0 if el <= 5000 else 1 for el in df[response_var]]
    elif dataset == 'letter':
        fn = '{}/letter.csv.gz'.format(utilmlab.get_data_dir())
    elif dataset == 'letter-recognition':
        fn = '{}/letter-recognition.csv.gz'.format(utilmlab.get_data_dir())
        labels = ['lettr']
    elif dataset == 'creditcardfraud':
        fn = ds2fn_d[dataset]
        labels = ['Amount', 'Class']
    elif dataset == 'show':
        print('availabel datasets: {}'.format(dataset_lst))
        sys.exit(0)
    else:
        print('warning: unsupported dataset:{}'.format(dataset))
        rval = 1
        return (rval, None)

    if df is not None:
        pass
    elif fn is not None:
        df = pd.read_csv(fn)
    else:
        assert 0

    features = [el for el in list(df.columns)
                if el not in labels and el not in features_drop]

    if nsample:
        df = df[:nsample]
    return (rval, {
        'df': df,
        'targets': labels,
        'features': features,
        'features_not_scalable': features_not_scalable
    } if not rval else None)


def dataset_log_properties(logger, dset):

    labels = dset['targets']

    df = dset['df']

    logger.info('features: #{} {}'.format(
        len(dset['features']), dset['features']))

    desc = ''
    for lbl in labels:
        card = len(set(df[lbl]))
        desc += str(card)

    logger.info('label(s): {} # {}'.format(labels, desc))

    logger.info('#: {}'.format(
        dset['df'].values.shape[0]))


def load_dataset_from_csv(logger, icsv, label):
    logger.info('loading csv {}'.format(icsv))
    df = pd.read_csv(icsv)
    features = list(df.columns)
    if label is not None:
        assert label in features
        features.remove(label)
    logger.info('features: #{} {} label:{}'.format(
        len(features), features, label))
    rval = 0
    return (rval, {
        'df': df,
        'targets': [label],
        'features': features,
        'features_not_scalable': []
    } if not rval else None)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o')
    parser.add_argument('--dataset')
    parser.add_argument(
        '--target',
        default='label',
        help='targets, separated by ,')
    parser.add_argument('--pmiss', default=0, type=float)
    parser.add_argument('-n', default=0, type=int)
    parser.add_argument(
        '--showavailable',
        help='show available datasets',
        action='store_true')
    parser.add_argument(
        "--separator",
        default=',',
        help="separator to use when writing to csv file")
    return parser.parse_args()


if __name__ == '__main__':

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.setLevel(logging.INFO)

    args = init_arg()
    dataset = args.dataset
    fn_o = args.o
    label = args.target.split(',')
    p_miss = args.pmiss
    sep = args.separator
    nsample = args.n
    
    if args.showavailable:
        dataset_lst = get_available_datasets()
        logger.info('available datasets: {}'.format(dataset_lst))
        sys.exit(0)
    
    # hack: space marker: some tools cannot deal with spaces with are part of
    # the name
    dataset = dataset.replace('@', ' ') if dataset is not None else dataset

    rval, dset = get_dataset(dataset)
    assert rval == 0
    dataset_log_properties(logger, dset)
    features = dset['features']
    df = dset['df']
    logger.info(dset['targets'])
    assert len(dset['targets']) >= 1

    df_dst = df[features+dset['targets']]

    if label[0] not in dset['targets']:
        print(dset['targets'], label)
        print(df_dst.columns)
        df_dst[label] = df[dset['targets']]
        print(df.columns)
        df_dst = df_dst.drop(dset['targets'], axis=1)
        for el in dset['targets']:
            assert el not in df_dst.columns

    del df
    
    logger.info('{} {} o:{} lbl:{} pmiss:{}'.format(
        dataset,
        df_dst.values.shape,
        fn_o,
        label,
        p_miss))

    if nsample:
        df_dst = df_dst[:nsample]

    if p_miss:
        df_dst[features] = utilmlab.introduce_missing(df_dst[features], p_miss)
    assert fn_o is not None
    compression = 'gzip' if fn_o.endswith('.gz') else None
    logger.info('saving {} {}'.format(fn_o, list(df_dst.columns)))
    df_dst.to_csv(fn_o, index=False, compression=compression, sep=sep)
