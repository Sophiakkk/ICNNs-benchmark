import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import init
import math
from one_dim_funcs import grammy_and_lee
    
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
        z_3 = self.fc2_y(y)+F.linear(z_2, torch.exp(self.z2_W), None)
        return z_3
    

class one_dim_ICNNsTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_min: float,
                 x_max: float,
                 tmax: int,
                 method: str = 'ICNNs',
                 init_func_name: str = 'gramacy_and_lee',
                 num_epochs: int = 10000,
                 num_steps: int = 10000,
                 num_grids: int = 10000,
                 lr: float = 0.001,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.xmin = x_min
        self.xmax = x_max
        self.method = method
    
    def pick_init_func(self):
        if self.init_func_name == 'grammy_and_lee':
            self.init_func = grammy_and_lee
        else:
            assert False, "Invalid initial function name"

    def preprocess(self):
        x = np.linspace(self.xmin, self.xmax, self.num_grids)
        self.features = torch.tensor(x, requires_grad=True, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.u0 = torch.tensor(self.init_func(x), requires_grad=False, dtype=torch.float32).to(self.device) # generate the initial function values

    def train(self):
        for t in range(self.tmax+1):
            for epoch in range(self.num_epochs):
                self.optimizer.zero_grad()
                u = self.net(self.features)
                if t == 0:
                    loss = torch.mean(torch.square(u.squeeze()-self.u0))
                else:
                    loss = torch.mean(torch.square(u.squeeze()-ut.squeeze()))
                loss.backward()
                self.optimizer.step()
                # if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}")
            
            ut = torch.minimum(self.u0, u).detach()
            
            # Do GD
            x_opt = torch.tensor(np.random.uniform(self.xmin, self.xmax, size=(1,1)), requires_grad=True, dtype=torch.float32).to(self.device)
            print("x_initial is: ",x_opt)
            
            for j in range(self.num_steps):
                # Calculate the value of the function and its gradient
                y = self.net(x_opt)
                grad = torch.autograd.grad(y, x_opt, create_graph=True)[0]

                # Perform gradient descent update
                with torch.no_grad():
                    x_opt = (x_opt - self.lr * grad).clone().detach().requires_grad_(True)

            print("optima is: ",x_opt)
            final_opt = x_opt.clone().detach().cpu().numpy()
            u_x = torch.tensor(self.init_func(final_opt),dtype=torch.float32,requires_grad=False).to(self.device)
            f_x = self.net(x_opt).data

            if f_x < u_x:
                print("fx is: ",f_x)
                print("ux is: ",u_x)
                for k in range (self.num_epochs):
                    self.optimizer.zero_grad()

                    y_train = self.net(x_opt)
                    loss = torch.norm(y_train - u_x) # force the neural net learn the function
                    
                    # Backpropagation
                    loss.backward()
                    
                    # Update the model parameters
                    self.optimizer.step()

                    # Print the loss
                    if k % 1000 == 0:
                        print(f'Re-training Epoch [{k}/{self.num_epochs}], Loss: {loss.item():.8f}')
                        
            if t % 10 == 0:
                torch.save(self.net.state_dict(), "./models/{}_{}_T{}_t{}.pth".format(self.method, self.init_func_name, self.tmax,t))


class one_dim_ICNNs_Evaluator(object):
    def __init__(self,
                 net: nn.Module,
                 seed: int,
                 x_min: float,
                 x_max: float,
                 tmax: int,
                 x_opt: float,
                 method_name: str = 'ICNNs',
                 init_func_name: str = 'gramacy_and_lee',
                 total_iterations: int = 10000,
                 step_size: float = 0.001,
                 device: torch.device = torch.device("cpu")
                 ):
        self.net = net.eval()
        self.seed = seed
        self.init_func_name = init_func_name
        self.xmin = x_min
        self.xmax = x_max
        self.tmax = tmax
        self.total_iterations = total_iterations
        self.step_size = step_size
        self.x_opt = x_opt
        self.method_name = method_name
        self.device = device

    def initalizer(self):
        # Set the random seed
        np.random.seed(self.seed)
        self.initial_x = np.random.uniform(self.xmin, self.xmax, size=(1,1))

    def pick_init_func(self):
        if self.init_func_name == 'grammy_and_lee':
            self.init_func = grammy_and_lee
        else:
            assert False, "Invalid initial function name"

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
        errory = np.linalg.norm(self.init_func(x)- self.init_func(self.x_opt))
        with open("./results/{}_{}_T{}_eval.txt".format(self.method_name,self.init_func_name,self.tmax), "a") as f:
            f.write("seed {}: error (input) is {}, error (output) is {}\n".format(self.seed, errorx, errory))