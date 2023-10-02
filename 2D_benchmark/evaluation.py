import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *
import torch.optim as optim
import timeit

# Configuation
parser = argparse.ArgumentParser()
# parser.add_argument("-s", "--seed", type=int, default=0)
# parser.add_argument("-m","--method_name", type = str, default = "SLGD_d")
# parser.add_argument("-f","--func_name", type = str, default = "tray")
parser.add_argument("-n","--num_iter", type = int, default = 10000)
parser.add_argument("-beta", "--step_size", type = float, default = 0.001)
args = parser.parse_args()

# Parameters
seed_list = [1,2,3,4,5,6,7,8,9,10]
# func_list = ["ackley","bukin","dropwave","eggholder","griewank","langermann","levy","levy13",
#              "rastrigin","schaffer2","schwefel","tray", "holdertable", "schaffer4", "shubert"]
func_list = ["holdertable"]
# method_list = ["GD","SLGD_r","SLGD_d","autohomotopy","pinns"]
method_list = ["SLGD_d"]
total_iterations = args.num_iter
step_size = args.step_size

for method_name in method_list:
    if method_name == 'GD':
        for func_name in func_list:
            x_range = np.array(domain_range[func_name])
            x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
            with open("./results/{}_{}_eval.txt".format(method_name,func_name), "w") as f:
                f.write("Evaluation results for {} with {}:\n".format(method_name,func_name))
            for seed in seed_list:
                algorithm_evaluator = GDEvaluator(init_func_name = func_name, 
                                                seed=seed,
                                                x_range=x_range, 
                                                x_opt = x_opt,
                                                method_name = method_name)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.evaluate()

    elif method_name == 'SLGD_r':
        for func_name in func_list:
            x_range = np.array(domain_range[func_name])
            x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
            with open("./results/{}_{}_eval.txt".format(method_name,func_name), "w") as f:
                f.write("Evaluation results for {} with {}:\n".format(method_name,func_name))
            for seed in seed_list:
                algorithm_evaluator = SLGH_r_Evaluator(init_func_name = func_name, 
                                                seed=seed,
                                                x_range=x_range, 
                                                x_opt = x_opt,
                                                method_name = method_name,
                                                num_samples=50,
                                                tmax=50,)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.evaluate()

    elif method_name == 'SLGD_d':
        for func_name in func_list:
            x_range = np.array(domain_range[func_name])
            x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
            with open("./results/{}_{}_eval.txt".format(method_name,func_name), "w") as f:
                f.write("Evaluation results for {} with {}:\n".format(method_name,func_name))
            for seed in seed_list:
                algorithm_evaluator = SLGH_d_Evaluator(init_func_name = func_name, 
                                                seed=seed,
                                                x_range=x_range, 
                                                x_opt = x_opt,
                                                method_name = method_name,
                                                num_samples=50,
                                                tmax=50,)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.evaluate()
    
    elif method_name == 'autohomotopy':
        for func_name in func_list:
            x_range = np.array(domain_range[func_name])
            x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
            with open("./results/{}_{}_eval.txt".format(method_name,func_name), "w") as f:
                f.write("Evaluation results for {} with {}:\n".format(method_name,func_name))
            for seed in seed_list:
                eval_net = NeuralNet()
                eval_net.load_state_dict(torch.load("./models/{}_{}_T50_t50.pth".format(method_name,func_name),map_location=torch.device('cpu')))
                algorithm_evaluator = AutoHomotopy_Evaluator(net=eval_net,
                                                            x_range=x_range, 
                                                            tmax=50,
                                                            init_func_name = func_name, 
                                                            seed=seed,
                                                            x_opt = x_opt,
                                                            method_name = method_name,)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.evaluate()
    
    elif method_name == 'pinns':
        for func_name in func_list:
            x_range = np.array(domain_range[func_name])
            x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
            with open("./results/{}_{}_eval.txt".format(method_name,func_name), "w") as f:
                f.write("Evaluation results for {} with {}:\n".format(method_name,func_name))
            for seed in seed_list:
                eval_net = NeuralNet()
                eval_net.load_state_dict(torch.load("./models/{}_{}_T50.pth".format(method_name,func_name),map_location=torch.device('cpu')))
                algorithm_evaluator = PINNs_Evaluator(net=eval_net,
                                                            x_range=x_range, 
                                                            tmax=50,
                                                            init_func_name = func_name, 
                                                            seed=seed,
                                                            x_opt = x_opt,
                                                            method_name = method_name,)
                algorithm_evaluator.initalizer()
                algorithm_evaluator.evaluate()