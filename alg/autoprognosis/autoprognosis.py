import model
import argparse
import numpy as np
import json
import pandas as pd
import os
import pickle
import time
import initpath_ap
initpath_ap.init_sys_path()
import utilmlab


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", help="output directory")
    parser.add_argument(
        "-n",
        default=0,
        type=int,
        help="maximum number of samples")
    parser.add_argument(
        "--nstage",
        default=1,
        type=int,
        help=""
        "size of pipeline: 0: auto (selects imputation when missing data is detected)"
        "1: only classifiers, "
        "2: feature processesing + clf, "
        "3: imputers + feature processors and clf")
    parser.add_argument("--dataset")
    parser.add_argument("-i", help='input data in csv format')
    parser.add_argument(
        "--target",
        help='name of response var when using csv as input')
    parser.add_argument(
        "--separator",
        default=',',
        help="separator csv file")
    parser.add_argument("--metric", default='aucroc')
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument(
        "--pmiss",
        default=0,
        type=float,
        help='missing rate when introducing nans')
    parser.add_argument(
        "--xtr",
        default=0,
        type=int,
        help="private/adhoc/not supported parameter to"
        " temporally control internal behavior")
    parser.add_argument("--usegain", default=0, type=int)
    parser.add_argument(
        "--it",
        default=20,
        type=int,
        help='number of iterations')
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        help='include ensembles when fitting')
    parser.add_argument("--ensemblesize", default=3, type=int)
    parser.add_argument("--model", help="filename to save model")
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="number of cross validation folds")
    parser.add_argument(
        "--acquisitiontype",
        default='LCB',
        help="[LCB, MPI, EI], LCB is prefered but generates warnings")
    return parser.parse_args()


def impute_gain(x, odir):
    python_exe = 'python3'
    script = '{}/gain/gain.py'.format(utilmlab.get_proj_dir())
    fn_i = '{}/xmissing.csv.gz'.format(odir)
    fn_o = '{}/ximputed.csv'.format(odir)
    x.to_csv(fn_i, compression='gzip', index=False)
    if os.path.isfile(fn_o):
        os.remove(fn_o) 
    utilmlab.exe_cmd(
        logger,
        '{} {} -i {} -o {} --testall 1'.format(
            python_exe,
            script,
            fn_i,
            fn_o))
    return pd.read_csv(fn_o)


if __name__ == '__main__':
    version = '0.40'
    args = init_arg()
    nCV = args.cv
    model.set_xtr_arg(args.xtr)
    niter = args.it
    nsample = args.n
    use_gain = args.usegain
    odir = args.o if args.o is not None else '.'
    fn_json = '{}/result.json'.format(odir)
    fn_report_json = '{}/report.json'.format(odir)
    fn_model = args.model
    acquisition_type = args.acquisitiontype

    sep = args.separator
    verbose = args.verbose

    if not os.path.isdir(odir):
        print("error: output directory \"{}\" does not exist".format(odir))
        assert 0

    logger = utilmlab.init_logger(
        odir,
        'log_ap.txt',
        use_show=True if args.verbose else False)

    model.logger = logger
    metric = args.metric
    ds = args.dataset
    is_ensemble = args.ensemble
    ensemble_size = args.ensemblesize

    # hack: space marker: some tools cannot deal with spaces with are part of
    # the name
    ds = ds.replace('@', ' ') if ds is not None else ds
    
    nmax_model = args.nstage
    model.nmax_model = nmax_model
    fn_i = args.i
    label = args.target
    p_miss = args.pmiss

    logger.info('autoprognosis version'.format(version))
    logger.info('ds:{} stages:{} n:{} metric:{} pmiss:{} it:{} '
                'model:{} aqt:{}'.format(
                    ds, nmax_model, nsample,
                    metric, p_miss, niter, fn_model,
                    acquisition_type))

    logger.info('{}'.format(args))
    if ds is not None and os.path.isfile(
            '{}/util/datasets.py'.format(
                utilmlab.get_proj_dir())):
        import datasets
        X_, Y_ = datasets.load_dataset(ds)
    else:
        assert fn_i is not None
        assert label is not None
        logger.info('loading {} lbl:{} sep:{}'.format(
            fn_i, label, sep))
        df = pd.read_csv(fn_i, sep=sep)
        features = list(df.columns)
        assert label in features
        features.remove(label)
        assert len(df)
        assert len(features) > 1
        X_ = df[features]
        Y_ = df[label]

    if p_miss:
        X_ = pd.DataFrame(
            utilmlab.introduce_missing(X_, p_miss),
            columns=X_.columns)

    if nsample:
        nnan_x = utilmlab.df_get_num_na(X_)
        logger.info('+shape: x:{} y:{} nan: x:{} ({})'.format(
            X_.shape,
            Y_.shape,
            nnan_x,
            nnan_x/float(np.prod(X_.shape))
        ))
        X_ = X_.iloc[:nsample, :]
        Y_ = Y_[:nsample]

    nnan_x = utilmlab.df_get_num_na(X_)
    logger.info('shape: x:{} y:{} #{} nan: x:{} ({:0.3f})'.format(
        X_.shape,
        Y_.shape,
        len(set(Y_)),
        nnan_x,
        nnan_x/np.prod(X_.shape)))

    if nnan_x and use_gain:
        X_ = impute_gain(X_, odir)
        nnan_x = utilmlab.df_get_num_na(X_)
        logger.info('nan (after gain):{}'.format(nnan_x))

    # if nnan_x and 0 < nmax_model and nmax_model < 3:
    #     X_ = X_.fillna(X_.mean())
    #     logger.critical(
    #         'warning: nan (missing values) detected and no'
    #         'imputation selecting, using mean imputation:{}'.format(nnan_x))
    #     time.sleep(2)
    #     nnan_x = utilmlab.df_get_num_na(X_)
    #     assert nnan_x == 0

    exe_time_start = time.time()

    AP_mdl = model.AutoPrognosis_Classifier(
        CV=nCV,
        num_iter=niter,
        kernel_freq=100,
        ensemble=is_ensemble,
        ensemble_size=ensemble_size,
        Gibbs_iter=100,
        burn_in=50,
        num_components=3,
        metric=metric,
        isnan=True if utilmlab.df_get_num_na(X_) else False,
        acquisition_type=acquisition_type)

    if False:
        logger.info('+ap:evaluate_clf')
        Output, d = model.evaluate_clf(
            X_,
            Y_,
            AP_mdl,
            n_folds=nCV,
            visualize=True)
        logger.info('-ap:evaluate_clf')
        logger.info('+')
        for idx, el in enumerate(AP_mdl.scores_):
            logger.info('{} {:0.3f} {}'.format(
                idx, float(el), AP_mdl.eva_prop_[idx]))
        eva_prop_lst = sorted(
            AP_mdl.eva_prop_, key=lambda el: el[metric], reverse=True)
        logger.info(' -')
        for idx, el in enumerate(eva_prop_lst):
            logger.info('{} {}'.format(idx, el))
        logger.info('-')

        fn_result_clf = '{}/result_clf.json'.format(odir)
        with open(fn_result_clf, "w") as fp:
            json.dump({'result_clf': eva_prop_lst}, fp)

    assert is_ensemble == True

    Output, Output_ens, score_d, last_model_fitted, eva_prop_lst \
        = model.evaluate_ens(
            X_,
            Y_,
            AP_mdl,
            n_folds=nCV,
            visualize=False)

    AP_mdl = last_model_fitted

    if verbose:
        logger.info(' - Scores')
        for idx, el in enumerate(AP_mdl.scores_):
            logger.info('{} {:0.3f} {}'.format(
                idx, float(el), eva_prop_lst[-1][idx]))
        logger.info(' - sorted props')
        eva_prop_lst_last = sorted(
            eva_prop_lst[-1], key=lambda el: el[metric], reverse=True)
        for idx, el in enumerate(eva_prop_lst_last):
            logger.info('{} {}'.format(idx, el))
        logger.info('-')
    fn_result_clf = '{}/result_clf.json'.format(odir)
    with open(fn_result_clf, "w") as fp:
        json.dump({'result_clf': eva_prop_lst}, fp)
    
    score_d['time_exe'] = time.time() - exe_time_start

    with open(fn_json, "w") as fp:
        json.dump(score_d, fp)

    report_d = AP_mdl.APReport()
    report_d['classes'] = list(set(Y_))
    report_d['features'] = list(X_.columns)
    report_d['samples'] = len(Y_)

    with open(fn_report_json, "w") as fp:
        json.dump(report_d, fp)

    logger.debug('**Final Cross-validation score: ** {}'.format(str(Output)))
    logger.debug('**Final Cross-validation score with ensembles: ** {}'.format(
        str(Output_ens)))
    for ky in ['clf', 'clf_ens']:
        logger.debug(' evaluateAUC_ens: {} aucroc {:0.4f} #({})'.format(
            ky,
            np.mean(score_d[ky]['roc_lst']),
            len(score_d['clf']['roc_lst'])))
        logger.debug(' evaluateAUC_ens: {} aucprc {:0.4f} #({})'.format(
            ky,
            np.mean(score_d[ky]['prc_lst']),
            len(score_d[ky]['prc_lst'])))
    logger.info('-evaluateAUC_ens {} {} {:0.1f}s'.format(
        Output,
        Output_ens,
        score_d['time_exe']))

    if fn_model is not None:
        with open(fn_model, "wb") as fp:
            logger.info('saving model {}'.format(fn_model))
            pickle.dump(AP_mdl, fp)
