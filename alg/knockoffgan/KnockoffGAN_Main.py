'''
KnockoffGAN Generation Main Function
Jinsung Yoon (9/27/2018)
'''
import numpy as np
from tqdm import tqdm
from KnockoffGAN import KnockoffGAN
import os
import argparse
import copy
import logging
import time
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', default='.', help='input directory')
    parser.add_argument(
        '-o', default='.', help='output directory')
    parser.add_argument(
        '--bs', default=128, type=int)
    parser.add_argument(
        '-n', default=0, type=int)
    parser.add_argument(
        '--it', default=0, type=int)
    parser.add_argument(
        '--xname', default="Normal")
    parser.add_argument(
        '--yname', default="Logit")
    return parser.parse_args()


args = init_arg()

nsample = args.n
odir = args.o
idir = args.i
niter = args.it

logger = utilmlab.init_logger(
    odir,
    'log_knockoff.txt',
    log_level=logging.DEBUG)

logger.info('{}'.format(args))

mb_size = args.bs

#%% Settings
# 1. X Distribution
x_set = ['Normal','AR_Normal','Uniform','AR_Uniform']
x_name = x_set[0]
x_name = args.xname

# 2. Y|X Distribution
y_set = ['Logit','Gauss']
y_name = y_set[0]
y_name = args.yname

# Print Generating distribution 
logger.info('X: ' + x_name + ', Y: ' + y_name)

#%% Coefficients
if (x_name == 'AR_Normal') | (x_name == 'AR_Uniform'):
    Coef = np.asarray([0.0,0.1,0.2,0.3,0.4,0.6,0.8])
elif (x_name == 'Uniform') & (y_name == 'Logit'):
    Coef = np.asarray([1.0,2.0,3.0,4.0,5.0,6.0,7.0])
elif (x_name == 'Uniform') & (y_name == 'Gauss'):
    Coef = np.asarray([0.5,1.0,1.5,2.0,2.5,3.0,3.5])
elif (y_name == 'Logit'):
    Coef = np.asarray([5.0,6.0,7.0,8.0,9.0,10.0,11.0])
elif (y_name == 'Gauss'):
    Coef = np.asarray([2.0,2.5,3.0,3.5,4.0,4.5,5.0])
        
#%% Parameters
Replication = 100

#%% Iterations
for it in tqdm(range(Replication)):

    # For each coefficient
    for i in range(len(Coef)):

        # Select Coefficient
        amp_coef = Coef[i]

        # Read training data (the original data set (only features)
        file_name_X_src = '{}/Data/'.format(idir) + x_name + '_' + y_name + '/X_' + str(amp_coef) + '_' + str(it) + '.csv'
        file_name_X_dst = '{}/Knockoff_Data/'.format(idir) + x_name + '_' + y_name + '/X_' + str(amp_coef) + '_' + str(it) + '.csv'

        if os.path.isfile(file_name_X_dst):
            logger.debug('{} exist, skipping'.format(file_name_X_dst))
            continue
        logger.debug('{} ({})'.format(i, len(Coef)))
        logger.debug('reading {}'.format(file_name_X_src))
        x_train = np.loadtxt(file_name_X_src)
        logger.debug('x_train:'.format(x_train.shape))
        if nsample:
            x_train = x_train[:nsample, :]
        logger.debug('{},{} shape:{} {} bs:{}'.format(
            it,
            i,
            x_train.shape,
            np.prod(x_train.shape),
            mb_size))
        logger.debug('')

        time_start = time.time()

        # Generate Knockoff: generated data x~
        x_knockoff = KnockoffGAN(
            copy.deepcopy(x_train),
            x_name,
            mb_size=mb_size,
            niter=niter)

        # Write generated knockoff
        time_loop = time.time() - time_start
        time_left = time_loop*(Replication - it)*(len(Coef) - i)
        logger.debug('writing {} ({:0.0f}s) ({:0.0f}s)'.format(
            file_name_X_dst,
            time_loop,
            time_left/3600
        ))
        utilmlab.ensure_dir(os.path.dirname(file_name_X_dst))
        np.savetxt(file_name_X_dst, x_knockoff)
        del x_knockoff, x_train

        # to knockoff.R should take X, Y, and X~
