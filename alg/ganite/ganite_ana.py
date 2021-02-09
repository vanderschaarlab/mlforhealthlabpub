import argparse
import pandas as pd
import numpy as np
import json


def Perf_RPol_ATT(Test_T, Test_Y, Output_Y):
    # RPol
    # Decision of Output_Y
    hat_t = np.sign(Output_Y[:,1] - Output_Y[:,0])
    hat_t = (0.5*(hat_t + 1))
    new_hat_t = np.abs(1-hat_t)
    
    # Intersection
    idx1 = hat_t * Test_T
    idx0 = new_hat_t  * (1-Test_T)    

    # RPol Computation
    RPol1 = (np.sum(idx1 * Test_Y)/(np.sum(idx1)+1e-8)) * np.mean(hat_t)  
    RPol0 = (np.sum(idx0 * Test_Y)/(np.sum(idx0)+1e-8)) * np.mean(new_hat_t) 
    RPol = 1 - (RPol1 + RPol0)    
    
    # ATT   
    # Original ATT
    ATT_value = np.sum(Test_T * Test_Y) / (np.sum(Test_T) + 1e-8) - np.sum((1-Test_T) * Test_Y) / (np.sum(1-Test_T) + 1e-8)
    # Estimated ATT
    ATT_estimate = np.sum(Test_T * (Output_Y[:,1] - Output_Y[:,0]) ) / (np.sum(Test_T) + 1e-8) 
    # Final ATT    
    ATT = np.abs( ATT_value - ATT_estimate )
    return [RPol, ATT]


def PEHE(y, hat_y):
    return np.mean((np.square(
        ((y[:, 1]-y[:, 0])) - (hat_y[:, 1] - hat_y[:, 0]))))

def ATE(y, hat_y):
    return np.abs(np.mean(
        y[:, 1]-y[:, 0] ) - np.mean(hat_y[:, 1]-hat_y[:, 0]))


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parser.add_argument("--ref")
    parser.add_argument("--ref_treatment")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    fn_json = args.o
    
    df_hat = pd.read_csv(args.i)
    df_ref = pd.read_csv(args.ref)
    df_ref_treatment = None if args.ref_treatment is None \
                       else pd.read_csv(args.ref_treatment)
    y_hat = np.argmax(df_hat.values, axis=1)
    y_ref = np.argmax(df_ref.values, axis=1)

    result = dict()
    if df_ref_treatment is None:
        result['sqrt_PEHE'] = float(np.sqrt(
            PEHE(df_ref.values, df_hat.values)))
        result['ATE'] = float(ATE(df_ref.values, df_hat.values))
    else:
        result['Perf_RPol_ATT'] = float(Perf_RPol_ATT(
            df_ref_treatment.values,
            df_ref.values,
            df_hat.values)[0])
    print(result)
    with open(fn_json, "w") as fp:
        json.dump(result, fp)
