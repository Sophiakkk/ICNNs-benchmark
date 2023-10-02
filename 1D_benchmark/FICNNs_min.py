import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from one_dim_funcs import gramacy_and_lee, dyhotomy
    

class FICNNs(nn.Module):
    def __init__(self):
        super(FICNNs, self).__init__()
        self.fc0_y = nn.Linear(1, 32)
        self.fc1_y = nn.Linear(1, 32)
        self.fc2_y = nn.Linear(1, 1)

        self.fc1_z = nn.Linear(32, 32, bias=False)
        self.fc2_z = nn.Linear(32, 1, bias=False)
    
    def forward(self,y):
        z_1 = torch.relu(self.fc0_y(y))
        z_2 = torch.relu(self.fc1_y(y)+self.fc1_z(z_1))
        z_3 = self.fc2_y(y)+self.fc2_z(z_2)
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

        # clip the weights
        for name, param in model.named_parameters():
            # print(name, param.shape)
            if 'bias' not in name:
                if 'y' not in name:
                    param = param.data.clamp_(0,torch.inf)

        # Print the loss
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')
    
    ut = torch.minimum(u_initial,u).detach()
    if i % 5 == 0:
        plt.plot(x.detach().numpy(),u.detach().numpy(),label='FICNNs at t='+str(i))
# for name, param in model.named_parameters():
#         # print(name, param.shape)
#         if 'bias' not in name:
#             if 'y' not in name:
#                 print(param.data)

plt.plot(x.detach().numpy(),u_initial.detach().numpy(),label='Initial Func')
plt.legend()
plt.title("FICNNs with min")
plt.savefig("FICNNs_min.png")
plt.show()