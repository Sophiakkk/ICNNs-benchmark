import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *

eval_net = FICNNs()
eval_net.load_state_dict(torch.load("./models/ICNNs_ackley_T100_t0_lr0.001.pth",
                                            map_location=torch.device('cpu')))
eval_net.eval()
print(eval_net)
#print parameters
for name, param in eval_net.named_parameters():
    if name == 'z1_W' or name == 'z_last_W':
        print("exp", name,  torch.exp(param.data))
    if param.requires_grad:
        print(name, param.data)

# Test function
func_name = 'ackley'
x_range = np.array(domain_range[func_name])
x1 = np.linspace(x_range[0][0],x_range[0][1],100)
x2 = np.linspace(x_range[1][0],x_range[1][1],100)
X1, X2 = np.meshgrid(x1, x2)
features = torch.tensor(np.c_[X1.ravel(), X2.ravel()],dtype=torch.float32)
z = eval_net(features).detach().numpy().squeeze()
Z = z.reshape(100,100)
print(z.min())
print(z.max())

# # test plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X2,X1,Z,rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_xlim(x_range[1][1],x_range[1][0])
# ax.set_ylim(x_range[0][0],x_range[0][1])
# ax.set_zlim(z.min(),z.max())
# ax.set_xlabel('x2')
# ax.set_ylabel('x1')
# ax.set_title("ICNNs on {} at t = {} with lr{}".format(func_name,0,0.001))
# plt.show()
print(Z[0,:].min())
print(Z[0,:].max())
plt.ylim(Z[0,:].min(),Z[0,:].max())
plt.plot(X1[0,:],Z[0,:])
plt.show()

# Problem Resolved!
"""
The figure looks weird because of truncation errors. The Neural Net output is nearly the same for all inputs, and it is not perfectly accurate because of the truncation errors (
computed by float32). 
The truncation errors are accumulated during the forward propagation, and the Neural Net output is not accurate.
Therefore, the plot looks non-convex.
"""

# Why softplus is not good?
"""
The definition is softplus(x) = log(1 + exp(x)). Weirdly, the calculus derivation of softplus(x) is logsig(x) = 1.0 / (1.0 + exp(-x)).
The derivative of softplus(x) is always smaller than 1.0, this may lead to the vanishing gradient problem.
"""