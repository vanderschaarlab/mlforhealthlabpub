import numpy as np
import pandas as pd

def import_PHE_Stage3_version4(max_length=None):
    data = np.load('./data/phe_breast_cancer/stage3_event_sequence_version4.npz')
    
    event_list = ['E0', 'E11_ALL', 'E106', 'E16', 'E19&E20', 'E17', 'E18', 'E23', 'E27']
    
    feat_list  = ['AGE_GROUP', 'CT_FLAG', 'RT_FLAG', 'HT_FLAG', 'LATERALITY_L', 'LATERALITY_R',
                 'GRADE_1', 'GRADE_2', 'GRADE_3', 'SCREENDETECTED_0', 'SCREENDETECTED_1',
                 'MORPH_ICD10_O2_800', 'MORPH_ICD10_O2_801', 'MORPH_ICD10_O2_814', 'MORPH_ICD10_O2_821', 'MORPH_ICD10_O2_848',
                 'MORPH_ICD10_O2_850', 'MORPH_ICD10_O2_others', 'BEHAVIOUR_ICD10_O2_3','BEHAVIOUR_ICD10_O2_others',
                 'T_BEST_1', 'T_BEST_2', 'T_BEST_3', 'T_BEST_4', 'N_BEST_0', 'N_BEST_1', 'N_BEST_2', 'N_BEST_3',
                 'M_BEST_0', 'M_BEST_1', 'ER_STATUS_N', 'ER_STATUS_P', 'PR_STATUS_N', 'PR_STATUS_P',
                 'HER2_STATUS_N', 'HER2_STATUS_P','ETHNICITY_A', 'ETHNICITY_Others']
    
    label_180_list = ['LABEL_180', 'TEST_SEQ_IDX', 'E11_ALL']
    
    data_M = data['M']
    data_D = data['D']
    data_T = data['T']
    data_X = data['X']
    data_Y_180 = data['Y_180']

    if max_length is not None:
        data_M = data_M[:, :max_length, :]
        data_D = data_D[:, :max_length, :]
        data_T = data_T[:, :max_length, :]

    return data_M, data_D, data_T, data_X, data_Y_180, event_list, feat_list, label_180_list


def import_synthetic_HawkesProcess():
    tmp = np.load('./data/sample_HawkesProcess.npz')    
    data_M, data_D, data_T = tmp['data_M'], tmp['data_D'], tmp['data_T']
    
    return data_M, data_D, data_T