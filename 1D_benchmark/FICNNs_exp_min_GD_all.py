import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        z_3 = self.fc2_y(y)+F.linear(z_2, torch.exp(self.z2_W), None)
        return z_3

# Define the domain
x_min = -1
x_max = 3
t_max = 20
x = np.linspace(x_min, x_max, 1000)
num_points = 10000

# Initialize the neural network model and optimizer
model = FICNNs()
optimizer1 = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model.parameters(), lr=0.001)
x = torch.tensor(x, requires_grad=True, dtype=torch.float32).unsqueeze(1)
# Training loop
num_epochs = 10000
beta = 0.001
num_steps = 10000

u_initial = gramacy_and_lee(x.detach().numpy())
u_initial = torch.tensor(u_initial,requires_grad=False)

for i in range(t_max+1):
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer1.zero_grad()

        # Evaluate the model at the initial condition
        u = model(x)

        if i==0:
            loss = torch.mean((u - u_initial)**2, dim=0) # force the neural net learn the function
        else:
            loss = torch.mean((u - ut)**2, dim=0)
        
        # Backpropagation
        loss.backward()
        
        # Update the model parameters
        optimizer1.step()

        # Print the loss
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')

    if i % 5 == 0:
        plt.plot(x.detach().numpy(),u.detach().numpy(),label='FICNNs at t='+str(i))

    ut = torch.minimum(u_initial,u.detach())

    # Do GD
    x_opt = torch.tensor(np.random.uniform(x_min,x_max), requires_grad=False).expand(1,1)
    print("x_initial is: ",x_opt)
    
    for j in range(num_steps):
        x_new = x_opt.clone().requires_grad_(True)

        # Calculate the value of the function and its gradient
        y = model(x_new)
        grad = torch.autograd.grad(y, x_new, create_graph=True)[0]

        # Perform gradient descent update
        with torch.no_grad():
            x_opt = x_new - beta * grad
    print("optima is: ",x_opt)
    u_x = gramacy_and_lee(x_opt)
    f_x = model(x_opt.clone()).data
    
    x_train = torch.concatenate((x.clone().detach(),x_opt),0).requires_grad_(True)
    y_train_label = torch.cat((u.clone().detach(),u_x),0).requires_grad_(False)

    if f_x < u_x:
        print("fx is: ",f_x)
        print("ux is: ",u_x)
        for k in range (num_epochs):
            optimizer2.zero_grad()

            y_train = model(x_train)
            loss = torch.norm(y_train - y_train_label) # force the neural net learn the function
            
            # Backpropagation
            loss.backward()
            
            # Update the model parameters
            optimizer2.step()

            # Print the loss
            if k % 1000 == 0:
                print(f'Re-training Epoch [{k}/{num_epochs}], Loss: {loss.item():.8f}')

plt.plot(x.detach().numpy(),u_initial.detach().numpy(),label='Initial Func')
plt.legend()
plt.title("FICNNs with Exponential Weights and GD")
plt.savefig("./figures/FICNNs_exp_min_GD_all_T_{}.png".format(t_max))
plt.show()