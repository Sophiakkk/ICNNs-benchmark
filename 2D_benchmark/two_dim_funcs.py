import numpy as np
import matplotlib.pyplot as plt

# function list
function_list = ['ackley', 'holdertable', 'tray', 'bukin', 'dropwave', 
                 'eggholder', 'griewank', 'langermann', 'levy', 'levy13',
                 'rastrigin', 'schaffer2', 'schaffer4', 'schwefel', 'shubert']

# ackley function
def ackley(x,y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20

# holder table function
def holder_table(x,y):
    return -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)))

# cross in tray function
def cross_in_tray(x,y):
    return -0.0001*(np.abs(np.sin(x)*np.sin(y)*np.exp(np.abs(100 - np.sqrt(x**2 + y**2)/np.pi))) + 1)**0.1

# bukin function N. 6
def bukin(x,y):
    return 100*np.sqrt(np.abs(y - 0.01*x**2)) + 0.01*np.abs(x + 10)

#drop wave function
def drop_wave(x,y):
    return -(1 + np.cos(12*np.sqrt(x**2 + y**2)))/(0.5*(x**2 + y**2) + 2)

# eggholder function
def eggholder(x,y):
    return -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + (y + 47)))) - x*np.sin(np.sqrt(np.abs(x - (y + 47))))

# griewank function
def griewank(x,y):
    return 1 + (x**2 + y**2)/4000 - np.cos(x)*np.cos(y/np.sqrt(2))

# # langermann function (SFU)
# def langermann(x,y):
#     a = np.array([[3,5],[5,2],[2,1],[1,4],[7,9]])
#     c = np.array([1,2,5,2,3])
#     m = 5
#     result = 0
#     input = np.hstack((x,y)).reshape(-1,2)
#     for i in range(m):
#         result += c[i]*np.exp(-1/np.pi*np.sum((input - a[i,:])**2,axis=1))*np.cos(np.pi*np.sum((input - a[i,:])**2,axis=1))
#     return result

# langermann function (another paper)
def langermann(x,y):
    A = np.array([[9.681,0.667],[9.400,2.041],[8.025,9.152],[2.196,0.415],[8.074,8.777]])
    c = np.array([0.806,0.517,1.5,0.908,0.965])
    m = 5
    result = 0
    input = np.hstack((x,y)).reshape(-1,2)
    for i in range(m):
        result += -c[i]*np.exp(-1/np.pi*np.sum((input - A[i,:])**2, axis = 1))*np.cos(np.pi*np.sum((input - A[i,:])**2, axis = 1))
    return result

# levy function
def levy(x,y):
    return np.sin(np.pi*(1 + (x - 1)/4))*np.sin(2*np.pi*(1 + (y - 1)/4))*(1 + (x - 1)/4)**2 + (y - 1)**2*(1 + np.sin(2*np.pi*(1 + (y - 1)/4))**2)

# levy function N. 13
def levy_13(x,y):
    return np.sin(3*np.pi*x)**2 + (x - 1)**2*(1 + np.sin(3*np.pi*y)**2) + (y - 1)**2*(1 + np.sin(2*np.pi*y)**2)

# rastrigin function
def rastrigin(x,y):
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

# schaffer function N. 2
def schaffer_2(x,y):
    return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5)/(1 + 0.001*(x**2 + y**2))**2

# schaffer function N. 4
def schaffer_4(x,y):
    return 0.5 + (np.cos(np.sin(np.abs(x**2 - y**2)))**2 - 0.5)/(1 + 0.001*(x**2 + y**2))**2

# schwefel function
def schwefel(x,y):
    return 418.9829*2 - x*np.sin(np.sqrt(np.abs(x))) - y*np.sin(np.sqrt(np.abs(y)))

# shubert function
def shubert(x,y):
    sum1 = 0
    sum2 = 0
    for i in range(1,6):
        sum1 += i*np.cos((i + 1)*x + i)
        sum2 += i*np.cos((i + 1)*y + i)
    return sum1*sum2

# pick function among all the functions
def pick_function(function):
    if function == 'ackley':
        f0 = ackley
    elif function == 'bukin':
        f0 = bukin
    elif function == 'tray':
        f0 = cross_in_tray
    elif function == 'dropwave':
        f0 = drop_wave
    elif function == 'eggholder':
        f0 = eggholder
    elif function == 'griewank':
        f0 = griewank
    elif function == 'holdertable':
        f0 = holder_table
    elif function == 'langermann':
        f0 = langermann
    elif function == 'levy':
        f0 = levy
    elif function == 'levy13':
        f0 = levy_13
    elif function == 'rastrigin':
        f0 = rastrigin
    elif function == 'schaffer2':
        f0 = schaffer_2
    elif function == 'schaffer4':
        f0 = schaffer_4
    elif function == 'schwefel':
        f0 = schwefel
    elif function == 'shubert':
        f0 = shubert
    return f0

# x = np.linspace(0,10,100)
# y = np.linspace(0,10,100)
# X,Y = np.meshgrid(x,y)
# x_vec = X.ravel().reshape(-1,1)
# y_vec = Y.ravel().reshape(-1,1)
# print(x_vec.shape)
# input = np.hstack((x_vec,y_vec))
# print(input.shape)
# Z = langermann(x_vec,y_vec)
# Z = Z.reshape(X.shape)

# # test optimizer
# input = np.array([9.6810707,0.6666515])
# # input = np.array([2.00299219,1.006096])
# print(langermann(input[0],input[1]))

# # test plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlim(10,0)
# ax.plot_surface(X,Y,Z, rstride=1, cstride=1,
#                 cmap='viridis',edgecolor='none')
# plt.show()