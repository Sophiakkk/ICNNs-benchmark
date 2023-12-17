import argparse
import torch
import matplotlib.pyplot as plt
from Utility import *
from two_dim_funcs import *
from optimizers import *

parser = argparse.ArgumentParser()
parser.add_argument("-m","--method_name", type = str, default = "ICNNs")
parser.add_argument("-f","--func_name", type = str, default = "ackley")
parser.add_argument("-T", "--max_timestep", type = int, default = 100)
parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001)
args = parser.parse_args()

# Parameters
method_name = args.method_name
func_name = args.func_name
x_range = np.array(domain_range[func_name])
T = args.max_timestep
learning_rate = args.learning_rate

for k in range(T//10+1):
    t = k*10    
    model = FICNNs()
    model.load_state_dict(torch.load("./models/ICNNs_{}_T{}_t{}_lr{}.pth".format(func_name, T, t, learning_rate),
                                    map_location=torch.device('cpu')))
    model.eval()
    x_range = np.array(domain_range[func_name])
    x_opt = np.array(opt_solutions[func_name][0]) 
    x1 = np.linspace(x_range[0][0],x_range[0][1],100)
    x2 = np.linspace(x_range[1][0],x_range[1][1],100)
    X1, X2 = np.meshgrid(x1, x2)
    features = torch.tensor(np.c_[X1.ravel(), X2.ravel()],dtype=torch.float32)
    print(features.shape)
    z = model(features).detach().numpy().squeeze()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X2,X1,z.reshape(100,100),rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlim(x_range[1][1],x_range[1][0])
    ax.set_ylim(x_range[0][0],x_range[0][1])
    ax.set_xlabel('x2')
    ax.set_ylabel('x1')
    ax.set_title("ICNNs on {} at t = {} with lr{}".format(func_name,t,learning_rate))
    # plt.savefig("./figures/ICNNs_{}_T{}_t{}_lr{}.png".format(func_name,T,t,learning_rate))
    plt.show()