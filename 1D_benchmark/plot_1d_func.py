import numpy as np
import matplotlib.pyplot as plt
from one_dim_funcs import gramacy_and_lee

x = np.linspace(-1,3,1000)
y = gramacy_and_lee(x)

plt.plot(x,y)
plt.show()