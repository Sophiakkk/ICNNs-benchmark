import re
import numpy as np
import math

T_list = [100]
method_list = ['ICNNs']
func_list = ["ackley","bukin","tray","dropwave","eggholder","griewank","holdertable","langermann",
             "levy","levy13","rastrigin","schaffer2","schaffer4","schwefel", "shubert"]
seed_list = [1,2,3,4,5,6,7,8,9,10]
lr_list = [1e-3,1e-4,1e-5,1e-6,1e-7]

for tmax in T_list:
    for k in range(tmax//10+1):
        t = k*10
        for lr in lr_list:
            with open('summary_2d_T{}_t{}_lr{}.txt'.format(tmax,t,lr), 'w') as file:
                file.writelines('Summary for 2D benchmark with T={}, t={}, lr={}:\n'.format(tmax,t,lr))
            for method in method_list:
                for func in func_list:
                    metric_list_x = []
                    metric_list_y = []
                    with open('{}_{}_T{}_t{}_lr{}_eval.txt'.format(method, func, tmax,t, lr), 'r') as file:
                        for line in file:
                            if len(re.findall("\d+\.\d+", line))==2:
                                metric_list_x.append(float(re.findall("\d+\.\d+", line)[0]))
                                metric_list_y.append(float(re.findall("\d+\.\d+", line)[1]))
                        if len(metric_list_x) == 0:
                            avg_error_x = 'N/A'
                            std_error_x = 'N/A'
                            avg_error_y = 'N/A'
                            std_error_y = 'N/A'
                        else:
                            avg_error_x = '%.4g'%np.mean(metric_list_x)
                            std_error_x = '%.4g'%np.std(metric_list_x)
                            avg_error_y = '%.4g'%np.mean(metric_list_y)
                            std_error_y = '%.4g'%np.std(metric_list_y)
                    with open('summary_2d_T{}_t{}_lr{}.txt'.format(tmax,t,lr), 'a') as file:
                        file.writelines('Senario: ' + method + ' on function ' + func + '\n')
                        file.writelines('avg error (input): ' + avg_error_x + '\n')
                        file.writelines('std error (input): ' + std_error_x + '\n')
                        file.writelines('avg error (output): ' + avg_error_y + '\n')
                        file.writelines('std error (output): ' + std_error_y + '\n')
                        file.writelines('-----------------------------------------' + '\n')