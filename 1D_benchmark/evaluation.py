import torch
from one_dim_funcs import dyhotomy
import argparse
from utility import *

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-n","--num_iter", type = int, default = 10000)
parser.add_argument("-beta", "--step_size", type = float, default = 0.001)
parser.add_argument("-T", "--max_timestep", type = int, default = 50)
args = parser.parse_args()

# Parameters
seed_list = [1,2,3,4,5,6,7,8,9,10]
func_list = ["grammy_and_lee"]
# "langermann",
method_list = ["ICNNs"]
total_iterations = args.num_iter
step_size = args.step_size

for method_name in method_list:
    if method_name == 'ICNNs':
        for func_name in func_list:
            if func_name == 'grammy_and_lee':
                x_range_min = -1
                x_range_max = 3
                x_optimal = dyhotomy(0.5, 0.6, 0.0001)
            with open("./results/{}_{}_T{}_eval.txt".format(method_name,func_name,args.max_timestep), "w") as f:
                f.write("Evaluation results for {} with {} T {}:\n".format(method_name,func_name,args.max_timestep))
            for seed in seed_list:
                eval_net = FICNNs()
                eval_net.load_state_dict(torch.load("./models/{}_{}_T{}_t{}.pth".format(method_name,func_name,args.max_timestep,args.max_timestep),map_location=torch.device('cpu')))
                algorithm_evaluator = one_dim_ICNNs_Evaluator(net=eval_net,
                                                            x_min=x_range_min,
                                                            x_max=x_range_max,
                                                            tmax=args.max_timestep,
                                                            init_func_name = func_name, 
                                                            seed=seed,
                                                            x_opt=x_optimal)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.pick_init_func()
                algorithm_evaluator.evaluate()