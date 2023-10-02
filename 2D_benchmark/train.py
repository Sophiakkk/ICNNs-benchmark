import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *
import timeit

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-m","--method_name", type = str, default = "autohomotopy")
parser.add_argument("-f","--func_name", type = str, default = "ackley")
parser.add_argument("-T", "--max_timestep", type = int, default = 50)
args = parser.parse_args()

# Parameters
method_name = args.method_name
func_name = args.func_name
x_range = np.array(domain_range[func_name])
T = args.max_timestep

# Training loop
start = timeit.default_timer()

if method_name == 'autohomotopy':
    algorithm = AutoHomotopyTrainer(net=NeuralNet(), 
                                    x_range = x_range, 
                                    init_func_name=func_name, 
                                    method = method_name,
                                    tmax=T)
    algorithm.preprocess()
    algorithm.train()

elif method_name == 'pinns':
    algorithm = PINNsTrainer(net=NeuralNet(),
                            x_range = x_range,
                            init_func_name=func_name,
                            method = method_name,
                            tmax=T)
    algorithm.preprocess()
    algorithm.load_data()
    algorithm.train()

stop = timeit.default_timer()

with open ("./results/"+method_name+"_"+func_name+"_train.txt", "w") as f:
    f.write("Method: "+method_name+"\n")
    f.write("Function: "+func_name+"\n")
    f.write("Max Timestep: "+str(args.max_timestep)+"\n")
    f.write("Running Time: "+str(stop-start)+"\n")