import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from one_dim_funcs import gramacy_and_lee, dyhotomy
import numdifftools as nd
from torch.nn.parameter import Parameter
import init
import math

class FICNNs(nn.Module):
    def __init__(self):
        super(FICNNs, self).__init__()
        self.fc0_y = nn.Linear(1, 32)
        self.fc1_y = nn.Linear(1, 32)
        self.fc2_y = nn.Linear(1, 1)

        # Set weights and bias for z
        self.z1_W = Parameter(torch.empty((32, 32))).requires_grad_(True)
        init.kaiming_uniform_(self.z1_W, a=math.sqrt(5))
        self.z2_W = Parameter(torch.empty((1, 32))).requires_grad_(True)
        init.kaiming_uniform_(self.z2_W, a=math.sqrt(5))
    
    def forward(self,y):
        z_1 = torch.relu(self.fc0_y(y))
        z_2 = torch.relu(self.fc1_y(y)+F.linear(z_1, torch.exp(self.z1_W), None))
        z_3 = self.fc2_y(y)+F.linear(z_1, torch.exp(self.z2_W), None)
        return z_3
    
model = torch.load('./models/1d_benchmark_T60.pt')
model.eval()

# t = torch.tensor(50)
# x = torch.linspace(-2,2,1000).unsqueeze(1)
# print(x.shape)
# # Calculate the value of the function and its gradient
# y = model(torch.cat([t.expand_as(x), x],dim=1))
# plt.plot(x.squeeze(1).detach().numpy(), y.squeeze(1).detach().numpy())

SEED_list = [1,2,3,4,5,6,7,8,9,10]

total_iteration = 20000
gamma = 0.999
beta = 0.001
eta=0.001
t = 50
M = 50
final_result = []

# Define the starting point for optimization
t = torch.tensor([t])
# Define the learning rate for gradient descent
lr = torch.tensor(beta)

x_optimal = dyhotomy(0.5, 0.6, 0.0001)
x_min = -1
x_max = 3

# Optimization loop
for seed in SEED_list:
    np.random.seed(seed=seed)
    x_opt = torch.tensor(np.random.uniform(x_min,x_max), requires_grad=False).expand(1,1)
    print("x_initial is: ",x_opt)
    
    for i in range(total_iteration):
        x = x_opt.clone().requires_grad_(True)

        # Calculate the value of the function and its gradient
        y = model(x)
        grad = torch.autograd.grad(y, x, create_graph=True)[0]

        # Perform gradient descent update
        with torch.no_grad():
            x_opt = x - beta * grad
    print("final x is: ",x_opt)
    errorx = np.linalg.norm(x_opt-x_optimal)
    errory = np.linalg.norm(gramacy_and_lee(x_opt)- gramacy_and_lee(x_optimal))
    with open("./results/ICNNs_1d_eval.txt", "a") as f:
        f.write("seed {}: error (input) is {}, error (output) is {}\n".format(seed, errorx, errory))




# x_min = -2
# x_max = 2
# x = np.linspace(x_min, x_max, 1000)
# x = torch.tensor(x, requires_grad=False, dtype=torch.float32).unsqueeze(1)
# u = model(torch.cat([t.expand_as(x), x], dim=1))
# plt.plot(x.clone().squeeze(1).detach().numpy(), u.clone().squeeze(1).detach().numpy(), label='F(x,t=50)')
# plt.show()