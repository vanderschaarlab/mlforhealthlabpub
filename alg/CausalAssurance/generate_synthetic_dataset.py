import argparse

from causal_utils import *


# select your GPU Here
os.environ["CUDA_VISIBLE_DEVICES"]="" #Comment this line out if you want all GPUS 

dataset_path = '~/Desktop/Kaggle/Bike-Sharing-Dataset/hour.csv'

def init_arg():
    parser = argparse.ArgumentParser()
    

    parser.add_argument("-e", help = "Add an edge", 
                        type = str, 
                        nargs = '+', 
                        action = 'append')
    parser.add_argument("-n", type = int, default = 3, help = "number of nodes/variables")
    parser.add_argument("-m", type = int, default = 0, help = "mean")
    parser.add_argument("-v", type = int, default = 1, help = "variance")
    parser.add_argument("-s", type = int, default = 1000, help = "samples")
    parser.add_argument("-o", 
                        type = str, 
                        required = True,
                        help = "Output file")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()
    
    try:
        e = []
        for edge in args.e:
            e.append(tuple([int(edge[0]), int(edge[1])]))
        print(e)
    except:
        print("Error in edges")

    try:
        df = gen_data2(list_vertex = np.arange(args.n), 
                       list_edges = e,
                       mean = args.m, 
                       var = args.v, 
                       SIZE = args.s)
    except:
        print("Error in generating data: Check your edge are contained within the number of nodes in graph")
    
    df.to_csv(args.o)

    

    
