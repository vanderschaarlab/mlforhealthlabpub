# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pandas as pd
import numpy as np
import argparse
import logging


def init_sys_path():
    import os
    import sys
    proj_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir))
    sys.path.append(os.path.join(proj_dir, 'init'))
    import initpath
    initpath.platform_init_path(proj_dir)


init_sys_path()
import utilmlab


logger = logging.getLogger()


def intergrise(f):
    return [int(f[k]) for k in range(len(f))]


dataset_names = ['UNOS_waitlist','UNOS_post_transplant','MAGGIC','SUPPORT','BPD','SEER_Breast_cancer_mortality',
                 'SEER_Leukemia_cancer_mortality','SEER_Respiratory_cancer_mortality','SEER_Digestive_cancer_mortality',
                 'SEER_Male_cancer_mortality','SEER_Female_cancer_mortality','SEER_Urinary_cancer_mortality',
                 'SEER_all_mortality','SEER_Breast_cancer','SEER_Leukemia_cancer','SEER_Respiratory_cancer',
                 'SEER_Digestive_cancer','SEER_Male_cancer','SEER_Female_cancer','SEER_Urinary_cancer',
                 'CF_mortality','CF_Liver disease','CF_Asthma','CF_ABPA','CF_Hypertension','CF_Diabetes','CF_Arthropathy',
                 'CF_Bone Fracture','CF_Osteoporosis','CF_Osteopenia','CF_Cancer','CF_Cirrhosis','CF_Kidney stones',
                 'CF_Depression','CF_Pseudo','CF_Pancreatitus','CF_Hearing loss','CF_Gall bladder',
                 'CF_Intestinal obstruction','CF_Staph']


def load_dataset(name):

    data_dir = utilmlab.get_data_dir()
    
    SEER_mortality = ['SEER_Breast_cancer_mortality','SEER_Leukemia_cancer_mortality',
                      'SEER_Respiratory_cancer_mortality','SEER_Digestive_cancer_mortality',
                      'SEER_Male_cancer_mortality','SEER_Female_cancer_mortality','SEER_Urinary_cancer_mortality']
    
    SEER_cause     = ['SEER_Breast_cancer','SEER_Leukemia_cancer',
                      'SEER_Respiratory_cancer','SEER_Digestive_cancer',
                      'SEER_Male_cancer','SEER_Female_cancer','SEER_Urinary_cancer']
    
    CF_names       = ['CF_mortality','CF_Liver disease','CF_Asthma','CF_ABPA','CF_Hypertension','CF_Diabetes','CF_Arthropathy',
                      'CF_Bone Fracture','CF_Osteoporosis','CF_Osteopenia','CF_Cancer','CF_Cirrhosis','CF_Kidney stones',
                      'CF_Depression','CF_Pseudo','CF_Pancreatitus','CF_Hearing loss','CF_Gall bladder',
                      'CF_Intestinal obstruction','CF_Staph'] 
    
    real_CF_names  = ['Liver Disease','Asthma','ABPA','Hypertension','Diabetes','Arthropathy','Bone fracture',
                      'Osteoporosis','Osteopenia','Cancer','Cirrhosis','Kidney Stones','Depression','Pseudomonas Aeruginosa',
                      'Pancreatitus','Hearing Loss','Gall bladder','Intestinal Obstruction','Staphylococcus Aureus']
    
    cmpl_CF_names  = ['cmpl_liver_disease','cmpl_asthma','cmpl_abpa','cmpl_hypertension','cmpl_CFRD','cmpl_arthropathy',
                      'cmpl_bone_fracture','cmpl_osteoporosis','cmpl_osteopenia','cmpl_cancer','cmpl_cirrhosis','cmpl_kidneystones',
                      'cmpl_depression','cult_pseudo_aeruginosa','cmpl_pancreatitus','cmpl_hearing_loss','cmpl_gall_bladder',
                      'cmpl_intestinal_obs','cult_staph'] 
    
    causes         = [intergrise(['26000']),intergrise(['35011','35012','35013','35021','35031','35022','35023','35041','35043']),
                      intergrise(['22010','22020','22030','22050','22060']),intergrise(['21010','21020','21030','21040','21050','21060','21071','21072','21080','21090','21100','21110','21120','21130']),
                      intergrise(['28010','28020','28030','28040']),intergrise(['27010','27020','27030','27040','27050','27060','27070']),
                      intergrise(['29010','29020','29030','29040'])]
    
    causes         = [intergrise(['50130']),intergrise(['35011','35012','35013','35021','35031','35022','35023','35041','35043']),
                      intergrise(['22010','22020','22030','22050','22060']),intergrise(['21010','21020','21030','21040','21050','21060','21071','21072','21080','21090','21100','21110','21120','21130']),
                      intergrise(['28010','28020','28030','28040']),intergrise(['27010','27020','27030','27040','27050','27060','27070']),
                      intergrise(['29010','29020','29030','29040'])] 
    
    if name in CF_names:
        
        if name != 'CF_mortality':
            
            Targets       = pd.read_csv('{}/CF2015.csv'.format(data_dir),encoding='latin', low_memory=False) 
            Data_in       = pd.read_csv('{}/CF_data.csv'.format(data_dir),encoding='latin', low_memory=False)
            Data_in.drop(['Label'],axis=1,inplace=True)
            
            CF_key_  = np.where(np.array([(CF_names[k] == name)*1 for k in range(len(CF_names))])==1)[0][0]
            IDs_pos_ = Targets.loc[Targets[cmpl_CF_names[CF_key_-1]].isin(['SELECT','selected']),'ID']

            Data_in['Labels_']                                  = 0
            Data_in.loc[Data_in['ID'].isin(IDs_pos_),'Labels_'] = 1
            Data_out                                            = Data_in[Data_in[real_CF_names[CF_key_-1]]==0].copy(True)
            Data_out.drop([real_CF_names[CF_key_-1]],axis=1,inplace=True)
            Data_out.drop(['Unnamed: 0', 'ID'],axis=1,inplace=True)
            
            X_      = Data_out.drop(['Labels_'],axis=1,inplace=False) 
            Y_label = Data_out['Labels_'] 
            ages    = Data_out['Age']
            survs   = 0
            
        else:  
            
            Data_in = pd.read_csv('{}/CF_data.csv'.format(data_dir),encoding='latin')
            
            X_      = Data_in.drop(['Label','Unnamed: 0', 'ID'],axis=1,inplace=False) 
            Y_label = Data_in['Label'] 
            ages    = Data_in['Age']
            survs   = 0    
    
    elif name == 'UNOS_waitlist':
        
        Dataset = pd.read_csv('{}/UNOS_waitlist.csv'.format(data_dir))
        Dataset.drop(["'Trans Year'"],axis=1,inplace=True)

        X       = Dataset.drop(["'Censor'","'Survival'"],axis=1,inplace=False)
        Y       = Dataset[["'Censor'","'Survival'"]]

        X       = X[~(Y["'Survival'"]==0)]
        Y       = Y[~(Y["'Survival'"]==0)]
        
        X_      = X[~((Y["'Survival'"]<365*3) & (Y["'Censor'"]==0))]
        Y_      = Y[~((Y["'Survival'"]<365*3) & (Y["'Censor'"]==0))]
        Y_label = ((Y_["'Survival'"]<365*3) & (Y_["'Censor'"]==1))*1
        
        ages    = X["'Age'"]
        survs   = Y[~((Y["'Survival'"]<365*3) & (Y["'Censor'"]==0))]["'Survival'"]/365
        
    elif name == 'UNOS_post_transplant': 
        
        Dataset = pd.read_csv('{}/UNOS_posttransplant.csv'.format(data_dir)) 
        Dataset.drop(["'Trans Year'"],axis=1,inplace=True)

        X       = Dataset.drop(["'Censor'","'Survival'"],axis=1,inplace=False)
        Y       = Dataset[["'Censor'","'Survival'"]]

        X       = X[~(Y["'Survival'"]==0)]
        Y       = Y[~(Y["'Survival'"]==0)]
        
        X_      = X[~((Y["'Survival'"]<365*3) & (Y["'Censor'"]==0))]
        Y_      = Y[~((Y["'Survival'"]<365*3) & (Y["'Censor'"]==0))]
        Y_label = ((Y_["'Survival'"]<365*3) & (Y_["'Censor'"]==1))*1
        
        ages    = X["'Age'"]
        survs   = Y[~((Y["'Survival'"]<365*3) & (Y["'Censor'"]==0))]["'Survival'"]/365        

    elif name == 'MAGGIC': 
        
        Dataset = pd.read_csv('{}/MAGGIC_data.csv'.format(data_dir))
        XX      = Dataset.drop(['Unnamed: 0','death_all','days_to_fu','maggic_id'], axis=1)
        YY      = Dataset[['days_to_fu','death_all']]
        X       = XX[(~YY['days_to_fu'].isnull()) & (YY['days_to_fu']>0)]
        Y       = YY[(~YY['days_to_fu'].isnull()) & (YY['days_to_fu']>0)]
        X_      = X[~((Y['days_to_fu']<365*3) & (Y['death_all']==0))]
        Y_      = Y[~((Y['days_to_fu']<365*3) & (Y['death_all']==0))]
        Y_label = ((Y_['days_to_fu']<365*3) & (Y_['death_all']==1))*1

        ages    = X["age"]
        survs   = Y[~((Y['days_to_fu']<365*3) & (Y['death_all']==0))]['days_to_fu']/365 
        
    elif name == 'SUPPORT':
        
        Data    = pd.read_csv('{}/support_data.csv'.format(data_dir))
        X       = Data[['sex','white','black','asian','hispanic','num.co','diabetes','dementia','Cancer','meanbp','hrt','resp',
                        'temp','wblc','crea','sod','age']]
        Y       = Data[['d.time','death']]
        X_      = X[~((Y['d.time']<365*3) & (Y['death']==0))]
        Y_      = Y[~((Y['d.time']<365*3) & (Y['death']==0))]
        Y_label = ((Y_['d.time']<365*3) & (Y_['death']==1))*1
        
        ages    = X["age"]
        survs   = Y[~((Y['d.time']<365*3) & (Y['death']==0))]['d.time']/365 
        
    elif name == 'BPD':    
        
        Dataset = pd.read_csv('{}/BPD_Data.csv'.format(data_dir))

        X       = Dataset.drop(['death_all', 'days_to_fu'],axis=1,inplace=False)
        Y       = Dataset[['death_all', 'days_to_fu']]
        X       = X[~(Y['days_to_fu']==0)]
        Y       = Y[~(Y['days_to_fu']==0)]
        X_      = X[~((Y['days_to_fu']<365*1) & (Y['death_all']==0))]
        Y_l     = Y[~((Y['days_to_fu']<365*1) & (Y['death_all']==0))]
        Y_label = ((Y_l['days_to_fu']<365*1) & (Y_l['death_all']==1))*1
        
        ages    = X["age_start"]
        survs   = Y[~((Y['days_to_fu']<365*3) & (Y['death_all']==0))]['days_to_fu']/365   
        
    elif name in SEER_mortality:
    
        SEERData     = pd.read_csv('{}/SEER_allcause_data.csv'.format(data_dir))
        Data         = SEERData[SEERData['CohortID'].isin([SEER_mortality.index(name)+1])]
        X            = Data[['Age', 'White', 'Black', 'Hispanic', 'Coarse Tumor Size', 'Bilateral', 'Single', 
                             'Married','Separated', 'Divorced', 'Widowed', 'Surgery', 'Benign','Borderline Malignancy', 
                             'Carcinoma In Situ', 'Malignant','Infiltrating Duct Carcinoma', 'Domestic Partner']]
        Y            = Data[['Survival Months','Censoring']]
        X_           = X[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]
        Y_           = Y[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]
        Y_label      = ((Y_['Survival Months']<12*10) & (Y_['Censoring']==1))*1
        
        ages         = X["Age"]
        survs        = Y[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]['Survival Months']/12 
        
    elif name in SEER_cause:
    
        SEERData       = pd.read_csv('{}/SEER_allcause_data.csv'.format(data_dir))
        Data           = SEERData[SEERData['CohortID'].isin([SEER_cause.index(name)+1])]
        X              = Data[['Age', 'White', 'Black', 'Hispanic', 'Coarse Tumor Size', 'Bilateral', 'Single', 
                               'Married','Separated', 'Divorced', 'Widowed', 'Surgery', 'Benign','Borderline Malignancy', 
                               'Carcinoma In Situ', 'Malignant','Infiltrating Duct Carcinoma', 'Domestic Partner']]
        Y              = Data[['Survival Months','Censoring']].copy(True)
        Y['Censoring'] = (Data['Cause'].isin(causes[SEER_cause.index('SEER_Breast_cancer')]))*1
        
        
        X_           = X[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]
        Y_           = Y[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]
        Y_label      = ((Y_['Survival Months']<12*10) & (Y_['Censoring']==1))*1
        
        ages         = X["Age"]
        survs        = Y[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]['Survival Months']/12     
        
        
    elif name == 'SEER_all_mortality':
        
        Data         = pd.read_csv('{}/SEER_allcause_data.csv'.format(data_dir))
        X            = Data[['Age', 'White', 'Black', 'Hispanic', 'Coarse Tumor Size', 'Bilateral', 'Single', 
                             'Married','Separated', 'Divorced', 'Widowed', 'Surgery', 'Benign','Borderline Malignancy', 
                             'Carcinoma In Situ', 'Malignant','Infiltrating Duct Carcinoma', 'Domestic Partner']].copy(True)
        Y            = Data[['Survival Months','Censoring']]
        
        X['Breast_cancer']      = (Data['CohortID']==1)*1 
        X['Leukemia_cancer']    = (Data['CohortID']==2)*1 
        X['Respiratory_cancer'] = (Data['CohortID']==3)*1 
        X['digestive_cancer']   = (Data['CohortID']==4)*1 
        X['Male_cancer']        = (Data['CohortID']==5)*1 
        X['Female_cancer']      = (Data['CohortID']==6)*1 
        X['Urinary_cancer']     = (Data['CohortID']==7)*1 

        X_           = X[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]
        Y_           = Y[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]
        Y_label      = ((Y_['Survival Months']<12*10) & (Y_['Censoring']==1))*1
        
        ages         = X["Age"]
        survs        = Y[~((Y['Survival Months']<12*10) & (Y['Censoring']==0))]['Survival Months']/12
    else:
        if logger is not None:
            logger.info('error: dataset:{} not available'.format(name))
        assert 0

    if logger is not None:
        logger.info('nan x:{} {} y:{}'.format(
            sum(np.ravel(np.isnan(X_))),
            sum(np.ravel(np.isnan(X_)))/len(np.ravel(X_)),sum(np.isnan(Y_label))))
    return X_, Y_label


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o")
    parser.add_argument("--dataset", default='BPD')
    parser.add_argument("--target", default='label')
    parser.add_argument("--pmiss", default=0, type=float)
    parser.add_argument(
        "--separator",
        default=',',
        help="separator to use when writing to csv file")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()
    dataset = args.dataset
    fn_o = args.o
    label = args.target
    p_miss = args.pmiss
    sep = args.separator
    
    # hack: space marker: some tools cannot deal with spaces with are part of
    # the name
    dataset = dataset.replace('@', ' ') if dataset is not None else dataset
    
    x, y = load_dataset(dataset)
    print('{} {} {} o:{} lbl:{} pmiss:{}'.format(dataset, x.shape, y.shape, fn_o, label, p_miss))
    if p_miss:
        x = utilmlab.introduce_missing(x, p_miss),
    df = pd.DataFrame(x)
    assert label not in df.columns
    df[label] = y
    assert fn_o is not None
    compression = 'gzip' if fn_o.endswith('.gz') else None
    df.to_csv(
        fn_o,
        index=False,
        compression=compression,
        sep=sep)
