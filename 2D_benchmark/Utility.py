import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from two_dim_funcs import *
from torch.nn.parameter import Parameter
import init
import math


# your algorithm is class(object):
#     def __init__(self):
#         self.net = NeuralNet()
#      
#     def train(self, x, y):
#         # do something....
#     
#     def preprocess(self, x, t):
#         # do something....

class FICNNs(nn.Module):
    def __init__(self):
        super(FICNNs, self).__init__()
        self.fc0_y = nn.Linear(2, 32)
        self.fc1_y = nn.Linear(2, 32)
        self.fc_last_y = nn.Linear(2, 1)

        # Set weights and bias for z
        self.z1_W = Parameter(torch.empty((32, 32))).requires_grad_(True)
        init.kaiming_uniform_(self.z1_W, a=math.sqrt(5))
        self.z_last_W = Parameter(torch.empty((1, 32))).requires_grad_(True)
        init.kaiming_uniform_(self.z_last_W, a=math.sqrt(5))
    
    def forward(self,y):
        z_1 = torch.relu(self.fc0_y(y))
        z_2 = torch.relu(self.fc1_y(y)+F.linear(z_1, torch.exp(self.z1_W), None))
        z_final = self.fc_last_y(y)+F.linear(z_2, torch.exp(self.z_last_W), None)
        return z_final
    
# class FICNNs(nn.Module):
#     def __init__(self):
#         super(FICNNs, self).__init__()
#         self.fc0_y = nn.Linear(2, 32)
#         self.fc1_y = nn.Linear(2, 32)
#         self.fc2_y = nn.Linear(2, 32)
#         self.fc_last_y = nn.Linear(2, 1)

#         # Set weights and bias for z
#         self.z1_W = Parameter(torch.empty((32, 32))).requires_grad_(True)
#         init.kaiming_uniform_(self.z1_W, a=math.sqrt(5))
#         self.z2_W = Parameter(torch.empty((32, 32))).requires_grad_(True)
#         init.kaiming_uniform_(self.z2_W, a=math.sqrt(5))
#         self.z_last_W = Parameter(torch.empty((1, 32))).requires_grad_(True)
#         init.kaiming_uniform_(self.z_last_W, a=math.sqrt(5))
    
#     def forward(self,y):
#         z_1 = torch.relu(self.fc0_y(y))
#         z_2 = torch.relu(self.fc1_y(y)+F.linear(z_1, torch.exp(self.z1_W), None))
#         z_3 = torch.relu(self.fc2_y(y)+F.linear(z_2, torch.exp(self.z2_W), None))
#         z_final = self.fc_last_y(y)+F.linear(z_3, torch.exp(self.z_last_W), None)
#         return z_final
    
class ICNNsTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 init_func_name: str,
                 method: str,
                 tmax: int,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 num_epochs: int = 10000,
                 num_steps: int = 10000,
                 num_grids: int = 100,
                 lr: float = 0.001,
                 ):
        self.tmax = tmax        # the maximum value of t
        self.lr = lr
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.device = device
        self.init_func_name = init_func_name
        self.num_grids = num_grids      # the number of grids in each dimension
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.x_range = x_range
        self.method = method
    
    def preprocess(self):
        self.xmin = self.x_range[:,0]
        self.xmax = self.x_range[:,1]
        x1 = np.linspace(self.xmin[0], self.xmax[0], self.num_grids)
        x2 = np.linspace(self.xmin[1], self.xmax[1], self.num_grids)
        X1, X2 = np.meshgrid(x1, x2)
        features = np.c_[X1.ravel(), X2.ravel()]
        self.init_func = pick_function(self.init_func_name) # pick the initial function w.r.t. the function name
        self.features = torch.tensor(features, requires_grad=True, dtype=torch.float32).to(self.device)
        self.u0 = torch.tensor(self.init_func(X1.ravel().reshape(-1,1),X2.ravel().reshape(-1,1)), requires_grad=False, dtype=torch.float32).to(self.device) # generate the initial function values

    def train(self):
        for t in range(self.tmax+1):
            for epoch in range(self.num_epochs):
                self.optimizer.zero_grad()
                u = self.net(self.features)
                if t == 0:
                    loss = torch.mean(torch.square(u-self.u0),dim=0)
                else:
                    u_local_min = self.net(x_local_min)
                    loss = torch.mean(torch.maximum(u-self.u0,torch.tensor(0).to(self.device)),dim=0) + torch.maximum(u0_x_local_min-u_local_min,torch.tensor(0).to(self.device))
                loss.backward()
                self.optimizer.step()
                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}")
            
            # Do GD
            x_opt = torch.tensor(np.random.uniform(self.xmin, self.xmax, size=(1,2)), requires_grad=False, dtype=torch.float32).to(self.device)
            print("x_initial is: ",x_opt)
            
            for j in range(self.num_steps):
                x_new = x_opt.clone().requires_grad_(True).to(self.device)

                # Calculate the value of the function and its gradient
                y = self.net(x_new)
                grad = torch.autograd.grad(y, x_new, create_graph=True)[0]

                # Perform gradient descent update
                with torch.no_grad():
                    x_opt = x_new - self.lr * grad

            print("optima is: ",x_opt)
            final_opt = x_opt.clone().detach().cpu().numpy()
            u0_x_local_min = torch.tensor(self.init_func(final_opt[:,0],final_opt[:,1]),dtype=torch.float32,requires_grad=False).to(self.device)
            x_local_min = x_opt.clone().requires_grad_(True).to(self.device) # only local optimizer
                        
            if t % 10 == 0:
                torch.save(self.net.state_dict(), "./models/{}_{}_T{}_t{}_lr{}.pth".format(self.method, self.init_func_name, self.tmax,t,self.lr))


# class ICNNsTrainer(object):
#     def __init__(self,
#                  net: nn.Module,
#                  x_range: np.ndarray,
#                  init_func_name: str,
#                  method: str,
#                  tmax: int,
#                  device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#                  num_epochs: int = 10000,
#                  num_steps: int = 10000,
#                  num_grids: int = 100,
#                  lr: float = 0.001,
#                  ):
#         self.tmax = tmax        # the maximum value of t
#         self.lr = lr
#         self.net = net.to(device)
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
#         self.device = device
#         self.init_func_name = init_func_name
#         self.num_grids = num_grids      # the number of grids in each dimension
#         self.num_epochs = num_epochs
#         self.num_steps = num_steps
#         self.x_range = x_range
#         self.method = method
    
#     def preprocess(self):
#         self.xmin = self.x_range[:,0]
#         self.xmax = self.x_range[:,1]
#         x1 = np.linspace(self.xmin[0], self.xmax[0], self.num_grids)
#         x2 = np.linspace(self.xmin[1], self.xmax[1], self.num_grids)
#         X1, X2 = np.meshgrid(x1, x2)
#         features = np.c_[X1.ravel(), X2.ravel()]
#         self.init_func = pick_function(self.init_func_name) # pick the initial function w.r.t. the function name
#         self.features = torch.tensor(features, requires_grad=True, dtype=torch.float32).to(self.device)
#         self.u0 = torch.tensor(self.init_func(X1.ravel().reshape(-1,1),X2.ravel().reshape(-1,1)), requires_grad=False, dtype=torch.float32).to(self.device) # generate the initial function values

#     def train(self):
#         for t in range(self.tmax+1):
#             for epoch in range(self.num_epochs):
#                 self.optimizer.zero_grad()
#                 u = self.net(self.features)
#                 if t == 0:
#                     loss = torch.mean(torch.square(u-self.u0),dim=0)
#                 else:
#                     loss = torch.mean(torch.square(u-ut),dim=0)
#                 loss.backward()
#                 self.optimizer.step()
#                 if epoch % 1000 == 0:
#                     print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}")
            
#             ut = torch.minimum(self.u0, u).clone().detach()
            
#             # Do GD
#             x_opt = torch.tensor(np.random.uniform(self.xmin, self.xmax, size=(1,2)), requires_grad=False, dtype=torch.float32).to(self.device)
#             print("x_initial is: ",x_opt)
            
#             for j in range(self.num_steps):
#                 x_new = x_opt.clone().requires_grad_(True).to(self.device)

#                 # Calculate the value of the function and its gradient
#                 y = self.net(x_new)
#                 grad = torch.autograd.grad(y, x_new, create_graph=True)[0]

#                 # Perform gradient descent update
#                 with torch.no_grad():
#                     x_opt = x_new - self.lr * grad

#             print("optima is: ",x_opt)
#             final_opt = x_opt.clone().detach().cpu().numpy()
#             u_x = torch.tensor(self.init_func(final_opt[:,0],final_opt[:,1]),dtype=torch.float32,requires_grad=False).to(self.device)
#             f_x = self.net(x_opt).clone().detach().data
#             x_train = torch.cat((self.features,x_opt),dim=0).requires_grad_(True).to(self.device)
#             u_label = torch.cat((ut,u_x.reshape(1,1)),dim=0).requires_grad_(False).to(self.device)
#             # x_train = x_opt.clone().requires_grad_(True).to(self.device)

#             if f_x < u_x:
#                 print("fx is: ",f_x)
#                 print("ux is: ",u_x)
#                 for k in range (self.num_epochs):
#                     self.optimizer.zero_grad()
#                     y_train = self.net(x_train)
#                     # loss = torch.norm(y_train - u_x) # force the neural net learn the function
#                     loss = torch.mean(torch.square(y_train-u_label),dim=0)
#                     # Backpropagation
#                     loss.backward()
                    
#                     # Update the model parameters
#                     self.optimizer.step()

#                     # Print the loss
#                     if k % 1000 == 0:
#                         print(f'Re-training Epoch [{k}/{self.num_epochs}], Loss: {loss.item():.8f}')
                        
#             if t % 10 == 0:
#                 torch.save(self.net.state_dict(), "./models/{}_{}_T{}_t{}.pth".format(self.method, self.init_func_name, self.tmax,t))


class ICNNs_Evaluator(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 tmax: int,
                 init_func_name: str,
                 seed: int,
                 x_opt: np.ndarray,
                 method_name: str,
                 total_iterations: int = 10000,
                 step_size: float = 0.001,
                 ):
        self.net = net.eval()
        self.seed = seed
        self.init_func_name = init_func_name
        self.init_func = pick_function(init_func_name)
        self.xmin = x_range[:,0]
        self.xmax = x_range[:,1]
        self.tmax = tmax
        self.total_iterations = total_iterations
        self.step_size = step_size
        self.x_opt = x_opt
        self.method_name = method_name

    def initalizer(self):
        # Set the random seed
        np.random.seed(self.seed)
        self.initial_x = np.random.uniform(self.xmin, self.xmax, size=(1,2))

    def get_grad(self, x):
        input = x.clone().requires_grad_(True)
        u = self.net(input)
        grad_x = torch.autograd.grad(u, input)[0][0][:2].clone().detach()
        return grad_x

    def evaluate(self):
        x = torch.tensor(self.initial_x, requires_grad=False, dtype=torch.float32)
        # perform gradient descent
        for i in range(self.total_iterations):
            grad_x = self.get_grad(x)
            # grad_x = grad_x/np.linalg.norm(grad_x)
            x = x - self.step_size*grad_x
        errorx = np.linalg.norm(x-self.x_opt)
        errory = np.linalg.norm(self.init_func(x[0][0],x[0][1])- self.init_func(self.x_opt[0],self.x_opt[1]))
        with open("./results/{}_{}_T{}_eval.txt".format(self.method_name,self.init_func_name,self.tmax), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))