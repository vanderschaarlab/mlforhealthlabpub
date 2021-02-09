import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
#import tensorflow as tf
#Disable TensorFlow 2 behaviour
from sklearn.model_selection import KFold  
from sklearn.preprocessing import StandardScaler  
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import os
from sklearn.metrics import mean_squared_error
from CASTLE import CASTLE
from utils import random_dag, gen_data_nonlinear
from signal import signal, SIGINT
from sys import exit
import argparse
def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv')

    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument('--num_nodes', type = int, default = 10)
    
    parser.add_argument('--dataset_sz', type = int, default = 1000)
    parser.add_argument('--output_log', type = str, default = 'castle.log')
    parser.add_argument('--n_folds', type = int, default = 10)
    parser.add_argument('--reg_lambda', type = float, default = 1)
    parser.add_argument('--reg_beta', type = float, default = 5)
    parser.add_argument('--gpu', type = str, default = '')
    parser.add_argument('--ckpt_file', type = str, default = 'tmp.ckpt')
    parser.add_argument('--extension', type = str, default = '')
    parser.add_argument('--branchf', type = float, default = 4)
    
    
    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.random_dag:
        def swap_cols(df, a, b):
            df = df.rename(columns = {a : 'temp'})
            df = df.rename(columns = {b : a})
            return df.rename(columns = {'temp' : b})
        def swap_nodes(G, a, b):
            newG = nx.relabel_nodes(G, {a : 'temp'})
            newG = nx.relabel_nodes(newG, {b : a})
            return nx.relabel_nodes(newG, {'temp' : b})
        
        #Random DAG
        num_edges = int(args.num_nodes*args.branchf)
        G = random_dag(args.num_nodes, num_edges)

        noise = random.uniform(0.3, 1.0)
        print("Setting noise to ", noise)
        
        df = gen_data_nonlinear(G, SIZE = args.dataset_sz, var = noise).iloc[:args.dataset_sz]
        df_test =  gen_data_nonlinear(G, SIZE = int(args.dataset_sz*0.25), var = noise)
        
        for i in range(len(G.edges())):
            if len(list(G.predecessors(i))) > 0:
                df = swap_cols(df, str(0), str(i))
                df_test = swap_cols(df_test, str(0), str(i))
                G = swap_nodes(G, 0, i)
                break      
                
        #print("Number of parents of G", len(list(G.predecessors(i))))
        print("Edges = ", list(G.edges()))
        
    else:
        '''
        Toy DAG
        The node '0' is the target in the Toy DAG
        '''
        G = nx.DiGraph()
        for i in range(10):
            G.add_node(i)
        G.add_edge(1,2)
        G.add_edge(1,3)
        G.add_edge(1,4)
        G.add_edge(2,5)
        G.add_edge(2,0)
        G.add_edge(3,0)
        G.add_edge(3,6)
        G.add_edge(3,7)
        G.add_edge(6,9)
        G.add_edge(0,8)
        G.add_edge(0,9)

        if args.csv:
            df = pd.read_csv(args.csv)
            df_test = df.iloc[-1000:]
            df = df.iloc[:args.dataset_sz]
        else: 
            df = gen_data_nonlinear(G, SIZE = args.dataset_sz)
            df_test = gen_data_nonlinear(G, SIZE = 1000)
        
    scaler = StandardScaler()
    if args.random_dag:
        df = scaler.fit_transform(df)
    else:
        if args.csv:
            scaler.fit(pd.read_csv(args.csv))
            df = scaler.transform(df)
        else:
            df = scaler.fit_transform(df)
            
    df_test = scaler.transform(df_test)
  
    X_test = df_test
    y_test = df_test[:,0]
    X_DAG = df
    
    kf = KFold(n_splits = args.n_folds, random_state = 1)

    fold = 0
    REG_castle = []
    print("Dataset limits are", np.ptp(X_DAG), np.ptp(X_test), np.ptp(y_test))
    for train_idx, val_idx in kf.split(X_DAG):
        fold += 1
        print("fold = ", fold)
        print("******* Doing dataset size = ", args.dataset_sz , "****************")
        X_train = X_DAG[train_idx]
        y_train = np.expand_dims(X_DAG[train_idx][:,0], -1)
        X_val = X_DAG[val_idx]
        y_val = X_DAG[val_idx][:,0]

        w_threshold = 0.3
        castle = CASTLE(num_train = X_DAG.shape[0], num_inputs = X_DAG.shape[1], reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                            w_threshold = w_threshold, ckpt_file = args.ckpt_file)
        num_nodes = np.shape(X_DAG)[1]
        castle.fit(X_train, y_train, num_nodes, X_val, y_val, X_test, y_test)
        W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
        print(W_est)

        REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
        print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))

        if fold > 1:
            print(np.mean(REG_castle), np.std(REG_castle))

       
        ######################################################################
        ######################################################################
        ######Everything below this point is for logging purposes only.#######
        ######################################################################
        ######################################################################
        


    
    def format_str(mean, std):
        return "${0:.6f}".format(round(mean,6)) + " \pm {0:.6f}$    ".format(round(std,6))
    with open(args.output_log, "a") as logfile:
        logfile.write(str(args.dataset_sz) + ",  " + 
                    format_str(np.mean(REG_castle), np.std(REG_castle)) + 
                    '\n')
    print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle)) 


            
    
