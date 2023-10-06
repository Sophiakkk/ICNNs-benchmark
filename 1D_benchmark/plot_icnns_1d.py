import torch
import matplotlib.pyplot as plt
from utility import *

tmax = 50
num_grids = 10000
func_list = ["grammy_and_lee"]

model = FICNNs()
for func in func_list:
    fig = plt.figure()
    for j in range(tmax // 10 + 1):
        t = j * 10
        model.load_state_dict(torch.load("./models/ICNNs_{}_T{}_t{}.pth".format(func, tmax, t),
                                        map_location=torch.device('cpu')))
        model.eval()
        if func == 'grammy_and_lee':
            x_min = -1
            x_max = 3
        x = np.linspace(x_min,x_max,num_grids).reshape(-1,1)
        features = torch.tensor(x,dtype=torch.float32)
        y = model(features).detach().numpy().squeeze()        
        plt.plot(x, y, label = "ICNNs at t={}".format(t))
plt.xlim(x_min,x_max)
plt.legend()
plt.savefig("./figures/ICNNs_{}_T{}.png".format(func,tmax,t))