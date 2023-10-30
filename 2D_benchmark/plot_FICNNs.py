import torch
import matplotlib.pyplot as plt
from Utility import *
from two_dim_funcs import *
from optimizers import *

T_list = [100]
func_list = ["ackley","bukin","dropwave","eggholder","griewank",
            "levy","levy13","rastrigin","schaffer2","schwefel","shubert",
           "tray","holdertable","schaffer4"]
lr_list = [1e-3,1e-4,1e-5,1e-6,1e-7]

model = FICNNs()
for tmax in T_list:
    for k in range(tmax//10+1):
        t = k*10
        for func in func_list:
            for lr in lr_list:
                for j in range(tmax // 10 + 1):
                    t = (j) * 10
                    model.load_state_dict(torch.load("./models/ICNNs_{}_T{}_t{}_lr{}.pth".format(func, tmax, t, lr),
                                                    map_location=torch.device('cpu')))
                    model.eval()
                    x_range = np.array(domain_range[func])
                    x_opt = np.array(opt_solutions[func][0]) 

                    x1 = np.linspace(x_range[0][0],x_range[0][1],100)
                    x2 = np.linspace(x_range[1][0],x_range[1][1],100)
                    X1, X2 = np.meshgrid(x1, x2)
                    features = torch.tensor(np.c_[X1.ravel(), X2.ravel()],dtype=torch.float32)
                    print(features.shape)
                    z = model(features).detach().numpy().squeeze()
                    print(z.shape)
                    print(z.sum())

                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    # ax.plot_surface(xline.detach().numpy(),yline.detach().numpy(),z)
                    ax.plot_surface(X1,X2,z.reshape(100,100),rstride=1, cstride=1,
                                    cmap='viridis', edgecolor='none')
                    ax.set_xlim(x_range[0][0],x_range[0][1])
                    ax.set_ylim(x_range[1][0],x_range[1][1])
                    ax.set_title("ICNNs on {} at t = {} with lr{}".format(func,t,lr))
                    # ax.set_zlim(0, 25)
                    plt.savefig("./figures/ICNNs_{}_T{}_t{}_lr{}.png".format(func,tmax,t,lr))