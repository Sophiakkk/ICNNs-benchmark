import argparse
import timeit
from utility import *

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-m","--method_name", type = str, default = "ICNNs")
parser.add_argument("-f","--func_name", type = str, default = "grammy_and_lee")
parser.add_argument("-T", "--max_timestep", type = int, default = 100)
args = parser.parse_args()

# Parameters
method_name = args.method_name
func_name = args.func_name
T = args.max_timestep

# Training loop
start = timeit.default_timer()

if method_name == 'ICNNs':
    if func_name == 'grammy_and_lee':
        x_range_min = -1
        x_range_max = 3
        algorithm = one_dim_ICNNsTrainer(net=FICNNs(), 
                                        x_min= x_range_min,
                                        x_max = x_range_max, 
                                        init_func_name=func_name, 
                                        method = method_name,
                                        tmax=T)
        algorithm.pick_init_func()
        algorithm.preprocess()
        algorithm.train()

stop = timeit.default_timer()

with open ("./results/"+method_name+"_"+func_name+"T{}_train.txt".format(T), "w") as f:
    f.write("Method: "+method_name+"\n")
    f.write("Function: "+func_name+"\n")
    f.write("Max Timestep: "+str(args.max_timestep)+"\n")
    f.write("Running Time: "+str(stop-start)+"\n")