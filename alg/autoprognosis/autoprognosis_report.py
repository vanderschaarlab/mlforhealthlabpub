import numpy as np
import json
import argparse
from collections import defaultdict
import initpath_ap
initpath_ap.init_sys_path()
import utilmlab


def pretty_name(ky):
    pretty_name_d = {
        'best_score_single_clf': 'best score single clf (while fitting)',
        'best_score_single_pipeline': 'best score single pipeline (while fitting)',
        'ensemble_score': 'best ensemble score (while fittng)',
        'classes': 'classes dataset'
    }
    return pretty_name_d[ky] if ky in pretty_name_d.keys() else ky


def pretty_val(val):
    return '{:0.3f}'.format(val) if isinstance(val, float) else val


def generate_report(
        logger,
        score_d,
        report_d,
        clf_d,
        verbose):

    clf_pretty_name = {
        'clf': 'classifier',
        'clf_ens': 'ensemble'
    }
    score_pretty_name = {
        'roc_lst': 'aucroc',
        'prc_lst': 'aucprc'
    }
    logger.info('Score')
    logger.info('')
    for ky in ['clf', 'clf_ens']:
        for ky1 in ['roc_lst', 'prc_lst']:
            logger.info('{:15s} {} {:0.3f}'.format(
                clf_pretty_name[ky],
                score_pretty_name[ky1],
                np.mean(score_d[ky][ky1])))
    logger.info('')
    logger.info('Report')
    logger.info('')
    for ky in report_d.keys():
        if ky in ['kernel_members']:
            for ky1 in report_d[ky]:
                logger.info('{:45s} {} {}'.format(
                    pretty_name(ky),
                    pretty_name(ky1),
                    pretty_val(report_d[ky][ky1])))
        else:
            logger.info('{:45s} {}'.format(
                pretty_name(ky),
                pretty_val(report_d[ky])))

    eva_lst_all = list()
    for ky in clf_d.keys():
        for idx, lst in enumerate(clf_d[ky]):
            for el in lst:
                eva_lst_all.append((idx, el))

    opt_metric = report_d['optimisation_metric']
    logger.info(eva_lst_all[0])
    logger.info('sort by {}'.format(opt_metric))
    logger.info('# {}'.format(len(eva_lst_all)))
    lst = sorted(
        eva_lst_all, key=lambda el: el[1][opt_metric], reverse=True)
    clf_hyper_d = defaultdict(list)
    for el in lst:
        ky = 'hyperparameter_properties'
        if ky in el[1].keys():
            hyper_par = el[1][ky]
            model_par = str()
            for el1 in hyper_par:
                ky1 = 'hyperparameters'
                if ky1 in el1.keys():
                    hyper_par = el1[ky1]
                    model_par += hyper_par['model']
                else:
                    model_par += el1['name']
            if not verbose:
                # if not verbose group performance clf by name not by parameters
                model_par = el1['name']
            clf_hyper_d[model_par].append(el)

    logger.info('# {}'.format(len(clf_hyper_d.keys())))
    clf_lst = list()
    for el in clf_hyper_d.keys():
        # logger.info('{}'.format(el))
        aucroc_lst = list()
        aucprc_lst = list()
        for el1 in clf_hyper_d[el]:
            if 'initial' in el1[1]['name']:
                continue
            aucprc_lst.append(el1[1]['aucprc'])
            aucroc_lst.append(el1[1]['aucroc'])
        # logger.info('{:50s} {} {:0.3f} {:0.3f}'.format(el, len(aucroc_lst), np.mean(aucroc_lst), np.mean(aucprc_lst)))
        clf_lst.append((
            el,
            len(aucroc_lst),
            np.mean(aucroc_lst),
            np.mean(aucprc_lst)))

    sort_ky_num = 2 if opt_metric == 'aucroc' else 3
    
    clf_lst = sorted(
        clf_lst, key=lambda el: el[sort_ky_num], reverse=True)
    logger.info('')
    logger.info('Average performance per classifier (ignoring hyperparameters):')
    logger.info('')
    for idx, el in enumerate(clf_lst):
        logger.info('{:3d} {:50s} {:3d} {:0.3f} {:0.3f}'.format(idx, el[0], el[1], el[2], el[3]))
    logger.info('')
        
    if verbose:
        for el in lst[:verbose]:
            logger.info('{} {}'.format(el[0], el[1]))
    

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        help='autoprognosis output directory with results')
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='if not 0 show performance clf grouped by hyperparameters, otherwise grouped by clf name'
    )
    return parser.parse_args()


if '__main__' == __name__:
    args = init_arg()
    assert args.i is not None
    odir = args.i
    verbose = args.verbose
    logger = utilmlab.init_logger(odir, "log_report_ap.txt")
    fn_json = '{}/result.json'.format(odir)
    with open(fn_json, "r") as fp:
        score_d = json.load(fp)
    fn_json = '{}/result_clf.json'.format(odir)
    with open(fn_json, "r") as fp:
        clf_d = json.load(fp)
    fn_report_json = '{}/report.json'.format(odir)
    with open(fn_report_json, "r") as fp:
        report_d = json.load(fp)
    generate_report(
        logger,
        score_d,
        report_d,
        clf_d,
        verbose)
