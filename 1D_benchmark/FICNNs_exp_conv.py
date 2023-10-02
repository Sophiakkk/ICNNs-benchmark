import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from one_dim_funcs import gramacy_and_lee, dyhotomy
from torch.nn.parameter import Parameter
import init
import math
from scipy.ndimage import gaussian_filter
    

class FICNNs(nn.Module):
    def __init__(self):
        super(FICNNs, self).__init__()
        self.fc0_y = nn.Linear(1, 64)
        self.fc1_y = nn.Linear(1, 64)
        self.fc2_y = nn.Linear(1, 1)

        # Set weights and bias for z
        self.z1_W = Parameter(torch.empty((64, 64))).requires_grad_(True)
        init.kaiming_uniform_(self.z1_W, a=math.sqrt(5))
        self.z2_W = Parameter(torch.empty((1, 64))).requires_grad_(True)
        init.kaiming_uniform_(self.z2_W, a=math.sqrt(5))
    
    def forward(self,y):
        z_1 = torch.relu(self.fc0_y(y))
        z_2 = torch.relu(self.fc1_y(y)+F.linear(z_1, torch.exp(self.z1_W), None))
        z_3 = torch.relu(self.fc2_y(y)+F.linear(z_1, torch.exp(self.z2_W), None))
        return z_3

# Define the domain
x_min = -1
x_max = 3
t_max = 50
t_set = torch.linspace(0, t_max, t_max) # [0, 1, ..., 50]
x = torch.linspace(x_min, x_max, 1000)
num_points = 10000

# Initialize the neural network model and optimizer
model = FICNNs()
optimizer = optim.Adam(model.parameters(), lr=0.001)
x = torch.tensor(x, requires_grad=True, dtype=torch.float32).unsqueeze(1)
# Training loop
num_epochs = 10000

u_initial = gramacy_and_lee(x.detach().numpy())
u_initial = torch.tensor(u_initial,requires_grad=False)

for i in range(21):
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Evaluate the model at the initial condition
        u = model(x)

        if i==0:
            loss = torch.mean((u - u_initial)**2, dim=0) # force the neural net learn the function
        else:
            loss = torch.mean((u - ut)**2, dim=0)
        
        # Backpropagation
        loss.backward()
        
        # Update the model parameters
        optimizer.step()

        # Print the loss
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
    
    if i==0:
        ut = torch.minimum(u_initial,u).detach()
    else:
        conv_u = torch.tensor(gaussian_filter(ut.numpy(), mode="constant", sigma=0.1,cval=10))
        ut = torch.minimum(u_initial,conv_u).detach()
#    if i % 5 == 0:
plt.plot(x.detach().numpy(),u.detach().numpy(),label='FICNNs at t='+str(i))

# plt.plot(x.detach().numpy(),u_initial.detach().numpy(),label='Initial Func')
plt.plot(x.detach().numpy(),ut.numpy(),label='minimum Func')
plt.legend()
plt.title("Fully Input Convex Neural Networks")
plt.show()