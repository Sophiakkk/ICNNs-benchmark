import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import timeit
from one_dim_funcs import gramacy_and_lee, dyhotomy

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
x = torch.linspace(x_min, x_max, 1000)
num_points = 10000

u_initial = gramacy_and_lee(x.detach().numpy())
u_initial = torch.tensor(u_initial,requires_grad=False)

# Initialize the neural network model and optimizer
model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
x = torch.tensor(x, requires_grad=True, dtype=torch.float32).unsqueeze(1)
# Training loop
num_epochs = 10000

"""
    think about what to detach
"""

start = timeit.default_timer()
for i in range(t_max):
    if i == 0:
        ut = u_initial
    else:
        conv_fx_prime = torch.zeros_like(x)
        cov_t = torch.eye(x.shape[0])*2/(i+1)
        
        # For each t, sample once, and use the same sample for all x. This is for convergence.
        # If in each epoch, we recalculate the f'(x), then it will diverge; because in each epoch, the label is different.
        x_prime_set = torch.distributions.multivariate_normal.MultivariateNormal(loc=x.squeeze(), covariance_matrix=cov_t).sample((num_points,))
        for j in range(num_points):
            x_prime = x_prime_set[j].unsqueeze(1)
            fx_prime = model(torch.cat([t_set[i-1].expand_as(x_prime), x_prime], dim=1))
            conv_fx_prime += fx_prime
        conv_fx_prime = (conv_fx_prime/num_points).detach()
        fx = model(torch.cat([(t_set[i-1]).expand_as(x), x], dim=1)).detach()
        ut = torch.minimum(conv_fx_prime, fx)
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Evaluate the model at the initial condition
        u = model(torch.cat([(t).expand_as(x), x], dim=1))

        # Compute the loss using the PDE
        loss = torch.mean((u - ut)**2, dim=0) # force the neural net learn the function
        
        # Backpropagation
        loss.backward()
        
        # Update the model parameters
        optimizer.step()

        # Print the loss
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')

stop = timeit.default_timer()
print('Time: ', stop - start) 
torch.save(model, '../models/1D_Autohomotopy_t50.pt')

# # Evaluate the trained model at final time
# u_final = model(torch.cat([torch.full_like(x, t_max), x], dim=1))

# # Convert the torch tensors to numpy arrays
# x = x.squeeze(1).detach().numpy()
# u_final = u_final.squeeze(1).detach().numpy()

# # Plot the initial condition and the solution at final time
# plt.figure(figsize=(8, 6))
# plt.plot(x, u_initial, label='Initial Function')
# plt.plot(x, u_final, label='F(x,t={})'.format(t_max))
# plt.xlabel('x')
# plt.ylabel('u(t,x)')
# plt.legend()
# plt.title('Homotopy Solution using Neural Networks')
# plt.grid(True)
# plt.show()