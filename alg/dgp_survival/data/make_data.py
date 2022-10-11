# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pandas as pd


def load_SEER_data(fn_csv):

    train_size       = 50000
    
    PREDICT_features = ['AGE', 'TUMOURSIZE', 'NODESINVOLVED', 'ER_STATUS', 
                        'HER2_STATUS', 'SCREENDETECTED', 'GRADE']

    SEER_dataset     = pd.read_csv(fn_csv).sample(120000)

    feature_names    = PREDICT_features[0:5] + ['GRADE 1', 'GRADE 2'] 

    SEER_dataset['GRADE 1'] = (SEER_dataset['GRADE']==1)*1
    SEER_dataset['GRADE 2'] = (SEER_dataset['GRADE']==2)*1
    SEER_dataset['GRADE 3'] = (SEER_dataset['GRADE']==3)*1

    SEER_dataset            = SEER_dataset.loc[SEER_dataset['FOLLOW_UP_DAYS_DIAG_TO_VS'] > 0]
    
    X        = SEER_dataset[feature_names]
    T        = SEER_dataset['FOLLOW_UP_DAYS_DIAG_TO_VS'] 
    c        = (SEER_dataset['Coded Cause']==2)*1 + ((SEER_dataset['Coded Cause']==1) | (SEER_dataset['Coded Cause']==3))*2  

    all_data = SEER_dataset[feature_names + ['FOLLOW_UP_DAYS_DIAG_TO_VS'] + ['Coded Cause']]
    
    train_indexes = np.random.choice(list(range(len(SEER_dataset))), train_size, replace=False)
    test_indexes  = np.array(list(set(list(range(len(SEER_dataset)))) - set(train_indexes)))

    X_train = X.loc[X.index[train_indexes]]
    T_train = T.loc[T.index[train_indexes]]
    c_train = c.loc[c.index[train_indexes]]
    
    X_test  = X.loc[X.index[test_indexes]]
    T_test  = T.loc[T.index[test_indexes]]
    c_test  = c.loc[c.index[test_indexes]]

    return (X_train, T_train, c_train), (X_test, T_test, c_test) 
