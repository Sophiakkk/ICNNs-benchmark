import torch
import torch.nn as nn
import numpy as np
from two_dim_funcs import *
import numdifftools as nd
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
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

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # input features: 2(x) + 1(t) = 3
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # 1 output feature: u(t,x)

    def forward(self, inputs):
        x = torch.relu(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class test_NeuralNet(nn.Module):
    def __init__(self):
        super(test_NeuralNet, self).__init__()
        self.fcz1 = nn.Linear(2, 64)  
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 1 output feature: u(t,x)

    def forward(self, inputs):
        x = torch.relu(self.fcz1(inputs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AutoHomotopyTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 init_func_name: str,
                 method: str,
                 tmax: int = 50,
                 num_epochs: int = 10000,
                 num_samples: int = 50,
                 num_grids: int = 100,
                 lr: float = 0.001,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.tmax = tmax        # the maximum value of t
        self.lr = lr
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.device = device
        self.init_func_name = init_func_name
        self.num_grids = num_grids      # the number of grids in each dimension
        self.num_samples = num_samples      # the number of samples of each grid
        self.num_epochs = num_epochs
        self.x_range = x_range
        self.method = method
    
    def preprocess(self):
        xmin = self.x_range[:,0]
        xmax = self.x_range[:,1]
        x1 = np.linspace(xmin[0], xmax[0], self.num_grids)
        x2 = np.linspace(xmin[1], xmax[1], self.num_grids)
        X1, X2 = np.meshgrid(x1, x2)
        features = np.c_[X1.ravel(), X2.ravel()]
        init_func = pick_function(self.init_func_name) # pick the initial function w.r.t. the function name
        self.features = torch.tensor(features, requires_grad=True, dtype=torch.float32).to(self.device)
        self.u0 = torch.tensor(init_func(X1.ravel().reshape(-1,1),X2.ravel().reshape(-1,1)), requires_grad=False, dtype=torch.float32).to(self.device) # generate the initial function values

    def train(self):
        for t in range(self.tmax+1):
            t_vec = torch.tensor(np.repeat(t, self.num_grids**2).reshape(-1,1),requires_grad=True, dtype=torch.float32).to(self.device)
            input = torch.cat((self.features, t_vec), dim=1)
            cov_t = 2*t/(self.features.shape[1]+1) # the bandwidth is 2t/(d+1)
            for epoch in range(self.num_epochs):
                self.optimizer.zero_grad()
                u = self.net(input)
                if t == 0:
                    loss = torch.mean(torch.square(u-self.u0))
                else:
                    latent_prime = np.random.normal(loc=self.features.cpu().detach().numpy()[None,:],scale=cov_t,size=(self.num_samples,self.features.shape[0],self.features.shape[1]))
                    latent_prime = latent_prime.reshape(-1,self.features.shape[1])
                    t_prime = np.repeat(t-1, latent_prime.shape[0]).reshape(-1,1) # last time step
                    input_prime = torch.tensor(np.hstack([latent_prime,t_prime]),requires_grad=False,dtype=torch.float32).to(self.device)
                    # print(input_prime.shape) # (100000,3)
                    fx_prime = torch.mean(self.net(input_prime).reshape((self.num_samples,self.features.shape[0])),dim=0)
                    # print(fx_prime.shape)
                    # print(u.squeeze().shape)
                    loss = torch.mean(torch.square(u.squeeze()-fx_prime))
                loss.backward()
                self.optimizer.step()
                # if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}")
            if t % 10 == 0:
                torch.save(self.net.state_dict(), "./models/{}_{}_T{}_t{}.pth".format(self.method, self.init_func_name, self.tmax,t))

class PINNsTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 init_func_name: str,
                 method: str,
                 tmax: int = 50,
                 num_epochs: int = 10000,
                 num_grids: int = 100,
                 lr: float = 0.001,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.net = net
        self.tmax = tmax        # the maximum value of t
        self.lr = lr
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.device = device
        self.init_func_name = init_func_name
        self.num_grids = num_grids      # the number of grids in each dimension
        self.num_epochs = num_epochs
        self.x_range = x_range
        self.method = method
    
    def preprocess(self):
        xmin = self.x_range[:,0]
        xmax = self.x_range[:,1]
        x1 = np.linspace(xmin[0], xmax[0], self.num_grids)
        x2 = np.linspace(xmin[1], xmax[1], self.num_grids)
        x1_init, x2_init = np.meshgrid(x1, x2)
        t_vec = np.linspace(0, self.tmax, self.tmax+1)
        X1, X2, T = np.meshgrid(x1, x2, t_vec)
        self.init_func = pick_function(self.init_func_name) # pick the initial function w.r.t. the function name
        self.features = torch.tensor(np.c_[X1.ravel(), X2.ravel()],dtype=torch.float32)
        self.t = torch.tensor(T.ravel().reshape(-1,1), dtype=torch.float32)
        self.u0 = torch.tensor(self.init_func(X1.ravel().reshape(-1,1),X2.ravel().reshape(-1,1)), requires_grad=False, dtype=torch.float32) # generate the initial function values

    # Define the PDE function for each input feature
    def pde_loss(self, t, feature, u):
        # Compute the gradient of f
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True,retain_graph=True)[0] # Compute the time derivative of f
        du_dx = torch.autograd.grad(u, feature, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph= True)[0]
        
        # Compute the Hessian of f
        hessian = torch.zeros((feature.shape[1], feature.shape[1]))
        hessian[:,0] = torch.autograd.grad(du_dx[:,0].sum(), feature, create_graph=False, retain_graph= True)[0][0]
        hessian[:,1] = torch.autograd.grad(du_dx[:,1].sum(), feature, create_graph=False, retain_graph= True)[0][0]

        # Compute the minimum eigenvalue of the Hessian
        min_eig = torch.min(torch.linalg.eigvalsh(hessian))
        curvature = torch.min(torch.tensor(0), min_eig) # convexity condition

        # Compute the PDE
        pde_rhs = curvature * torch.sqrt(torch.tensor(1) + du_dx.norm()**2)
        pde_loss = (du_dt - pde_rhs).norm().detach()

        return pde_loss
    
    def load_data(self):
        dataset = TensorDataset(torch.cat((self.features, self.t), dim=1), self.u0)
        self.loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)

    # Training the PINNs model
    def train(self):
        for epoch in range(self.num_epochs):
            for batch_idx, (inputs, u0) in enumerate(self.loader):
                features = inputs[:,:2].to(self.device).requires_grad_(True)
                t = inputs[:,2].reshape(-1,1).to(self.device).requires_grad_(True)
                u0 = u0.to(self.device)

                self.optimizer.zero_grad()
                F_pred = self.net(torch.cat((features, t), dim=1))
                loss = self.pde_loss(t, features, F_pred) # pde(t, x, u)

                f0 = self.net(torch.cat((features, torch.zeros(inputs.shape[0]).reshape(-1,1).to(self.device)), dim=1))
                initial_condition_loss = torch.mean(torch.square(f0 - u0))
                loss += initial_condition_loss
                loss.backward()
                self.optimizer.step()

                # self.optimizer.zero_grad()
                # F_pred = self.net(torch.cat((self.features, self.t), dim=1))
                # loss = self.pde_loss(self.t, self.features, F_pred) # pde(t, x, u)
                # # Incorporate the boundary condition at t=0
                # initial_condition_loss = torch.mean(torch.square(F_pred.reshape((self.num_grids*self.num_grids, self.tmax+1))[:,0] - self.u0))
                # loss += initial_condition_loss
                # loss.backward()
                # self.optimizer.step()

            # if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}, Initial Condition Loss: {initial_condition_loss.item()}")
        
        torch.save(self.net.state_dict(), "./models/{}_{}_T{}.pth".format(self.method, self.init_func_name, self.tmax))

class FICNNsTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 init_func_name: str,
                 method: str,
                 num_epochs: int = 10000,
                 num_samples: int = 50,
                 num_grids: int = 100,
                 lr: float = 0.001,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.lr = lr
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.device = device
        self.init_func_name = init_func_name
        self.num_grids = num_grids      # the number of grids in each dimension
        self.num_samples = num_samples      # the number of samples of each grid
        self.num_epochs = num_epochs
        self.x_range = x_range
        self.method = method



class GDEvaluator(object):
    def __init__(self,
                 x_range: np.ndarray,
                 init_func_name: str,
                 seed: int,
                 x_opt: np.ndarray,
                 method_name: str,
                 total_iterations: int = 10000,
                 step_size: float = 0.001,
                 ):
        self.seed = seed
        self.init_func_name = init_func_name
        self.init_func = pick_function(init_func_name)
        self.xmin = x_range[:,0]
        self.xmax = x_range[:,1]
        self.total_iterations = total_iterations
        self.step_size = step_size
        self.x_opt = x_opt
        self.method_name = method_name
    
    def initalizer(self):
        # Set the random seed
        np.random.seed(self.seed)
        self.initial_x = np.random.uniform(self.xmin, self.xmax, size=(2,))

    def evaluate(self):
        x = self.initial_x
        # perform gradient descent
        for i in range(self.total_iterations):
            grad_x = nd.Gradient(self.init_func)(x[0],x[1])
            x = x - self.step_size*grad_x
        errorx = np.linalg.norm(x-self.x_opt)
        errory = np.linalg.norm(self.init_func(x[0],x[1])- self.init_func(self.x_opt[0],self.x_opt[1]))
        with open("./results/{}_{}_eval.txt".format(self.method_name,self.init_func_name), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))
        
class SLGH_r_Evaluator(object):
    def __init__(self,
                 x_range: np.ndarray,
                 tmax: int,
                 init_func_name: str,
                 seed: int,
                 x_opt: np.ndarray,
                 method_name: str,
                 num_samples, int = 50,
                 total_iterations: int = 10000,
                 step_size: float = 0.001,
                 discount_factor: float = 0.999,
                 ):
        self.seed = seed
        self.init_func_name = init_func_name
        self.init_func = pick_function(init_func_name)
        self.xmin = x_range[:,0]
        self.xmax = x_range[:,1]
        self.tmax = tmax
        self.total_iterations = total_iterations
        self.num_samples = num_samples
        self.step_size = step_size
        self.x_opt = x_opt
        self.discount_factor = discount_factor
        self.method_name = method_name
    
    def initalizer(self):
        # Set the random seed
        np.random.seed(self.seed)
        self.initial_x = np.random.uniform(self.xmin, self.xmax, size=(2,))

    def grad_estimate(self, t, x):
        d = x.shape[0]
        grad_x = 0
        grad_t = 0
        f_init = self.init_func(x[0],x[1])
        for i in range(self.num_samples):
            if t**2>0:
                v = np.random.normal(0,1,d)
                x_tmp = x+t*v
                f_tmp = self.init_func(x_tmp[0],x_tmp[1])
                # gradient estimate
                grad_x += 1/self.num_samples*v*(f_tmp-f_init)/t
                # gradient of t
                grad_t += 1/self.num_samples*(np.inner(v,v)-d)*(f_tmp-f_init)/t**2
            else:
                grad_x = nd.Gradient(self.init_func)(x[0],x[1])
        return grad_x, grad_t
    
    def evaluate(self):
        x = self.initial_x
        t = self.tmax            
        for i in range(self.total_iterations):
            grad_x, grad_t = self.grad_estimate(t,x)
            x = x - self.step_size*grad_x
            if t>0:
                t = self.discount_factor*t
        errorx = np.linalg.norm(x-self.x_opt)
        errory = np.linalg.norm(self.init_func(x[0],x[1])- self.init_func(self.x_opt[0],self.x_opt[1]))
        with open("./results/{}_{}_eval.txt".format(self.method_name,self.init_func_name), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))

class SLGH_d_Evaluator(object):
    def __init__(self,
                 x_range: np.ndarray,
                 tmax: int,
                 init_func_name: str,
                 seed: int,
                 x_opt: np.ndarray,
                 method_name: str,
                 num_samples: int = 50,
                 discount_factor: float = 0.999,
                 total_iterations: int = 10000,
                 step_size: float = 0.001,
                 threshold: float = 1e-3,
                 ):
        self.seed = seed
        self.init_func_name = init_func_name
        self.init_func = pick_function(init_func_name)
        self.xmin = x_range[:,0]
        self.xmax = x_range[:,1]
        self.tmax = tmax
        self.total_iterations = total_iterations
        self.num_samples = num_samples
        self.step_size = step_size
        self.x_opt = x_opt
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.method_name = method_name
    
    def initalizer(self):
        # Set the random seed
        np.random.seed(self.seed)
        self.initial_x = np.random.uniform(self.xmin, self.xmax, size=(2,))

    def grad_estimate(self, t, x):
        d = x.shape[0]
        grad_x = 0
        grad_t = 0
        f_init = self.init_func(x[0],x[1])
        for i in range(self.num_samples):
            if t**2>0:
                v = np.random.normal(0,1,d)
                x_tmp = x+t*v
                f_tmp = self.init_func(x_tmp[0],x_tmp[1])
                # gradient estimate
                grad_x += 1/self.num_samples*v*(f_tmp-f_init)/t
                # gradient of t
                grad_t += 1/self.num_samples*(np.inner(v,v)-d)*(f_tmp-f_init)/t**2
            elif math.isnan(x[0]) == False:
                grad_x = nd.Gradient(self.init_func)(x[0],x[1])
        return grad_x, grad_t

    def evaluate(self):
        x = self.initial_x
        t = self.tmax            
        for i in range(self.total_iterations):
            grad_x, grad_t = self.grad_estimate(t,x)
            x = x - self.step_size*grad_x
            if t>0:
                t = np.maximum(np.minimum(t*self.discount_factor,t-self.threshold*grad_t), 1e-10)
        errorx = np.linalg.norm(x-self.x_opt)
        errory = np.linalg.norm(self.init_func(x[0],x[1])- self.init_func(self.x_opt[0],self.x_opt[1]))
        with open("./results/{}_{}_eval.txt".format(self.method_name,self.init_func_name), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))

class PINNs_Evaluator(object):
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
        T = torch.tensor([[self.tmax]], dtype=torch.float32)
        input = torch.cat((x,T),1).requires_grad_(True)
        u = self.net(input)
        with torch.no_grad():
            grad_x = torch.autograd.grad(u, input)[0][0][:2]
        return grad_x

    def evaluate(self):
        x = torch.tensor(self.initial_x, dtype=torch.float32)
        # perform gradient descent
        for i in range(self.total_iterations):
            grad_x = self.get_grad(x)
            # grad_x = grad_x/np.linalg.norm(grad_x)
            x = x - self.step_size*grad_x
        errorx = np.linalg.norm(x-self.x_opt)
        errory = np.linalg.norm(self.init_func(x[0][0],x[0][1])- self.init_func(self.x_opt[0],self.x_opt[1]))
        with open("./results/{}_{}_eval.txt".format(self.method_name,self.init_func_name), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))

class AutoHomotopy_Evaluator(object):
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
        T = torch.tensor([[self.tmax]], dtype=torch.float32)
        input = torch.cat((x,T),1).requires_grad_(True)
        u = self.net(input)
        with torch.no_grad():
            grad_x = torch.autograd.grad(u, input)[0][0][:2]
        return grad_x

    def evaluate(self):
        x = torch.tensor(self.initial_x, dtype=torch.float32)
        # perform gradient descent
        for i in range(self.total_iterations):
            grad_x = self.get_grad(x)
            # grad_x = grad_x/np.linalg.norm(grad_x)
            x = x - self.step_size*grad_x
        errorx = np.linalg.norm(x-self.x_opt)
        errory = np.linalg.norm(self.init_func(x[0][0],x[0][1])- self.init_func(self.x_opt[0],self.x_opt[1]))
        with open("./results/{}_{}_eval.txt".format(self.method_name,self.init_func_name), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))