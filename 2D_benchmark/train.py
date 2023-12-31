import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *
import timeit

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-m","--method_name", type = str, default = "ICNNs")
parser.add_argument("-f","--func_name", type = str, default = "ackley")
parser.add_argument("-T", "--max_timestep", type = int, default = 50)
parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-6)
args = parser.parse_args()

# Parameters
method_name = args.method_name
func_name = args.func_name
x_range = np.array(domain_range[func_name])
T = args.max_timestep
learning_rate = args.learning_rate

# Training loop
start = timeit.default_timer()

if method_name == 'ICNNs':
    algorithm = ICNNsTrainer(net=FICNNs(), 
                                    x_range = x_range, 
                                    init_func_name=func_name, 
                                    method = method_name,
                                    tmax=T,
                                    lr=learning_rate)
    algorithm.preprocess()
    algorithm.train()

stop = timeit.default_timer()

with open ("./results/"+method_name+"_"+func_name+"T{}_lr{}_train.txt".format(T,learning_rate), "w") as f:
    f.write("Method: "+method_name+"\n")
    f.write("Function: "+func_name+"\n")
    f.write("Max Timestep: "+str(args.max_timestep)+"\n")
    f.write("Running Time: "+str(stop-start)+"\n")
