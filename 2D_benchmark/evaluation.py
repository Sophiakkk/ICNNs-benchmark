import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-n","--num_iter", type = int, default = 10000)
parser.add_argument("-beta", "--step_size", type = float, default = 0.001)
parser.add_argument("-T", "--max_timestep", type = int, default = 50)
args = parser.parse_args()

# Parameters
seed_list = [1,2,3,4,5,6,7,8,9,10]
func_list = ["ackley","bukin","dropwave","eggholder","griewank",
            "levy","levy13","rastrigin","schaffer2","schwefel","shubert",
           "tray","holdertable","schaffer4"]
# "langermann",
method_list = ["icnns"]
total_iterations = args.num_iter
step_size = args.step_size

for method_name in method_list:
    if method_name == 'icnns':
        for func_name in func_list:
            x_range = np.array(domain_range[func_name])
            x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
            with open("./results/{}_{}_T{}_eval.txt".format(method_name,func_name,args.max_timestep), "w") as f:
                f.write("Evaluation results for {} with {} T {}:\n".format(method_name,func_name,args.max_timestep))
            for seed in seed_list:
                eval_net = FICNNs()
                eval_net.load_state_dict(torch.load("./models/{}_{}_T{}_t{}.pth".format(method_name,func_name,args.max_timestep,args.max_timestep),map_location=torch.device('cpu')))
                algorithm_evaluator = ICNNs_Evaluator(net=eval_net,
                                                            x_range=x_range, 
                                                            tmax=args.max_timestep,
                                                            init_func_name = func_name, 
                                                            seed=seed,
                                                            x_opt = x_opt,
                                                            method_name = method_name)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.evaluate()