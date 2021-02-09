import numpy as np
import scipy


def init_output_dict(output, name=None):
    new = {}
    output[name] = new

    new['num_leaves'] = []
    new['within_var'] = []
    new['across_var'] = []
    new['PEHE'] = []
    new['CATE'] = {'coverage': [], 'intv': []}

    new['stats'] = {'num_leaves': {},
                    'within_var': {},
                    'across_var': {},
                    'PEHE': {},
                    'CATE': {}}

    return new


def update_output_dict(output, suboutput, result, name=None):
    suboutput['num_leaves'].append(result[5])
    suboutput['within_var'].append(result[6])
    suboutput['across_var'].append(result[7])
    suboutput['PEHE'].append(result[4])
    suboutput['CATE']['coverage'].append(result[3] * 100)
    suboutput['CATE']['intv'].append(np.mean(result[2][:, 1] - result[2][:, 0]))

    suboutput['stats']['num_leaves'] = mean_confidence_interval(suboutput['num_leaves'])
    suboutput['stats']['within_var'] = mean_confidence_interval(suboutput['within_var'])
    suboutput['stats']['across_var'] = mean_confidence_interval(suboutput['across_var'])
    suboutput['stats']['PEHE'] = mean_confidence_interval(suboutput['PEHE'])
    suboutput['stats']['CATE']['coverage'] = mean_confidence_interval(suboutput['CATE']['coverage'])
    suboutput['stats']['CATE']['intv'] = mean_confidence_interval(suboutput['CATE']['intv'])


def print_summary(suboutput, name="name"):
    print(f'| {name:>10} |'
          f' {suboutput["stats"]["across_var"][0]:6.2f} ± {suboutput["stats"]["across_var"][1]:6.2f} |'
          f' {suboutput["stats"]["within_var"][0]:6.2f} ± {suboutput["stats"]["within_var"][1]:6.2f} |'
          f' {suboutput["stats"]["num_leaves"][0]:6.2f} ± {suboutput["stats"]["num_leaves"][1]:6.2f} |'
          f' {suboutput["stats"]["CATE"]["intv"][0]:6.2f} ± {suboutput["stats"]["CATE"]["intv"][1]:6.2f} |')


def print_gain(output, output_root):
    gain = np.array(output["within_var"]) / np.array(output_root["within_var"])
    m, h = mean_confidence_interval(gain)

    print(f'Normalized V^in: {m:6.3f} ± {h:6.3f}')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h
