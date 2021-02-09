from keras.models import load_model
import pandas as pd
from sklearn.metrics import  mean_squared_error
from pycausal import prior as p
from sklearn.preprocessing import LabelEncoder
import argparse
import glob, os
from causal_utils import *


# select your GPU Here
os.environ["CUDA_VISIBLE_DEVICES"]="" #Comment this line out if you want all GPUS 

dataset_path = '~/Desktop/Kaggle/Bike-Sharing-Dataset/hour.csv'

def init_arg():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--modeldir", 
                        type = str,
                        #required = True,
                        help = "Path to assumed Keras (tensorflow) saved model")
    parser.add_argument("-e", help = "Add an edge", 
                        type = str, 
                        nargs = '+', 
                        action = 'append')
    parser.add_argument("-i", help = "Input data in csv format. This is the selection set csv.  It is assumed that columns are normalized appropriately")
    parser.add_argument("--discrete",
                        type = str,
                        action = 'append',
                        help = "List of discrete variables")
    parser.add_argument("--input_vars", 
                        type = str,
                        action = 'append',
                        #required = True,
                        help = "List of input variables in the form ['a', 'b', 'c']" )
    parser.add_argument("--target_var", 
                        type = str, 
                        #required = True,
                        help = "Target variable (single) as a string, i.e., 'a'")
    parser.add_argument("-l", 
                        type = float, 
                        default = 1,
                        help = "Lambda factor between validation metric and causal assurance term")
    parser.add_argument("-o", 
                        type = str, 
                        required = True,
                        help = "Output file")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()
    
    print(args.modeldir)
    filenames = []
    MSE = []
    LL = []
    for file in glob.glob(args.modeldir + '/*'):
        
        print(file)
        try:
            model = load_model(file)
        except:
            print("Could not load model:", file)
            assert(0)
        if model:
            print("Successfully loaded model")
        else:
            print("Error: Could not load model.  Please make sure you passed a  valid Keras(tensorflow) saved model")
            assert(0)
        
        # Read the target and input variables from the csv.
        try:
            print("Attempting to open input file:", args.i)
            input_df = pd.read_csv(args.i, usecols = args.input_vars) 
            target_df = pd.read_csv(args.i, usecols = [args.target_var])
            if len(input_df) > 0 and len(target_df) > 0:
                print("Successfully opened csvs with n-samples = ", len(input_df) )
            else:
                print("No samples found in csv")
                assert(0)
        except:
            print("Error: Could not open csv", args.i)
            assert(0)
    
        # Convert the discrete variables to categorical
        #Convert to labels
        label_encoder_list = []
        for i,col in enumerate(args.discrete):
            label_encoder_list.append(LabelEncoder())
            input_df[col] = label_encoder_list[i].fit_transform(input_df[col].values)
        
        # split into X,y
        if args.discrete:
            X = make_categorical(input_df, input_df, args.discrete)
        else:
            X = input_df.values
        y = target_df.values
    
        # Make prediction
        y_pred = model.predict(X)
        
        #Generate causal assured dataset D'
        causal_targets = pd.DataFrame(y_pred, columns = [args.target_var])
        causal_targets.reset_index(drop=True, inplace = True)
        causal_df = input_df.copy().join(causal_targets)
    
        
        # Calculate Score
        mse = mean_squared_error(y_pred, y)
        #print("MSE = ", mse)
        
        edges = list(map(tuple, args.e))
        #print("Required edges = ", edges)
        prior = p.knowledge(requiredirect= edges)
        if args.discrete:
            ll = get_ll_mixed(causal_df, prior)
        else:
            ll = get_ll_continuous(causal_df, prior)
        #print("LL(G|D) = ", ll)
        
        CAM = ll * args.l + mse
        #print("Overall CAM score = ", CAM)
        MSE.append(mse)
        LL.append(ll)
        filenames.append(file)
        # Save score
    combined = normalize(LL) * args.l + normalize(MSE)
    s = sorted(zip(combined, filenames))
    print(s)
    a, b = map(list, zip(*s))
    for cam, file in zip(a, b):
        with open(args.o, "a") as myfile:
            myfile.write(f'{file} {cam}\n')

    

    
