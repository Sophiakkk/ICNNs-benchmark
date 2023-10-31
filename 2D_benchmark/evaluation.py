import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-m","--method_name", type = str, default = "ICNNs")
parser.add_argument("-f","--func_name", type = str, default = "ackley")
parser.add_argument("-T", "--max_timestep", type = int, default = 50)
parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-6)
args = parser.parse_args()

# Parameters
seed_list = [1,2,3,4,5,6,7,8,9,10]
total_iterations = 10000
method_name = args.method_name
func_name = args.func_name
x_range = np.array(domain_range[func_name])
T = args.max_timestep
learning_rate = args.learning_rate

for k in range(T//10+1):
    t = k*10
    x_range = np.array(domain_range[func_name])
    x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
    with open("./results/{}_{}_T{}_t{}_lr{}_eval.txt".format(method_name,func_name,T,t,learning_rate), "w") as f:
        f.write("Evaluation results for {} with {} T {} t{} lr{}:\n".format(method_name,func_name,T,t,learning_rate))
    for seed in seed_list:
        eval_net = FICNNs()
        eval_net.load_state_dict(torch.load("./models/{}_{}_T{}_t{}_lr{}.pth".format(method_name,func_name,T,t,learning_rate),
                                            map_location=torch.device('cpu')))
        algorithm_evaluator = ICNNs_Evaluator(net=eval_net,
                                                    x_range=x_range, 
                                                    tmax=T,
                                                    t=t,
                                                    init_func_name = func_name, 
                                                    seed=seed,
                                                    x_opt = x_opt,
                                                    method_name = method_name,
                                                    lr = learning_rate)
        algorithm_evaluator.initalizer()
        algorithm_evaluator.evaluate()