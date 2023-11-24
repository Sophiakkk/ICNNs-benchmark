import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from two_dim_funcs import *
from Utility import *
from optimizers import *

func_name = "eggholder"
x_range = np.array(domain_range[func_name])
x_opt = np.array(opt_solutions[func_name][0])


xmin = x_range[:,0]
xmax = x_range[:,1]
num_grids = 100
x1 = np.linspace(xmin[0], xmax[0], num_grids)
x2 = np.linspace(xmin[1], xmax[1], num_grids)
X1, X2 = np.meshgrid(x1, x2)
features = np.c_[X1.ravel(), X2.ravel()]
init_func = pick_function(func_name) # pick the initial function w.r.t. the function name
features = torch.tensor(features, requires_grad=True, dtype=torch.float32)
u0 = torch.tensor(init_func(X1.ravel().reshape(-1,1),X2.ravel().reshape(-1,1)), requires_grad=False, dtype=torch.float32)

print("avg u0:",torch.mean(u0))
print("max u0:",torch.max(u0))
print("min u0:",torch.min(u0))
print("std u0:",torch.std(u0))
print(np.sort(u0.numpy(),axis=None)[:5000])
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X2,X1,u0.reshape(100,100),rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_ylim(x_range[0][0],x_range[0][1])
# ax.set_xlim(x_range[1][1],x_range[1][0])
# ax.set_title("Initial Function")
# ax.set_xlabel("x2")
# ax.set_ylabel("x1")
# plt.show()