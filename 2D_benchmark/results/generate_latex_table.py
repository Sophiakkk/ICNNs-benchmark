T_list = [100]
method_list = ['ICNNs']
func_list = ["ackley","bukin","tray","dropwave","eggholder","griewank","holdertable","langermann",
             "levy","levy13","rastrigin","schaffer2","schaffer4","schwefel", "shubert"]
seed_list = [1,2,3,4,5,6,7,8,9,10]
lr_list = [1e-3]
# lr_list = [1e-3,1e-4,1e-5,1e-6,1e-7]

with open("summary_all.txt", "r") as gdf:
    GD = {}
    for line in gdf:
        if line.startswith("Senario: GD"):
            func = line.split()[4]
            GD[func] = []
        if line.startswith("avg error (output)"):
            GD[func].append(line.split()[3])
        if line.startswith("std error (output)"):
            GD[func].append(line.split()[3]) 

data_list = [] 
for tmax in T_list:
    for k in range(tmax//20+1):
        t = k*20
        for lr in lr_list:
            with open ('summary_2d_T{}_t{}_lr{}.txt'.format(tmax,t,lr), "r") as f:
                data = {}
                for line in f:
                    if line.startswith("Senario"):
                        func = line.split()[4]
                        data[func] = []
                    if line.startswith("avg error (output)"):
                        data[func].append(line.split()[3])
                    if line.startswith("std error (output)"):
                        data[func].append(line.split()[3])
                data_list.append(data)

with open ("latex_summary.txt", "w") as file:
    file.write("\\begin{table}\n")
    file.write("\\centering\n")
    file.write("\\caption{")
    file.write("Benchmark Optimization Errors for lr{}".format(str(lr_list[0])))
    file.write("}\n")
    file.write("\\label{benchmark_yerrors_icnns}\n")
    file.write("\\resizebox{\\textwidth}{!}{\n")
    file.write("\\begin{tabular}{c|ccccccc}\n")
    file.write("\hline\n")
    file.write("Test Senerio & GD ")
    for j in range(len(data_list)):
        file.write("& ICNNs(t={})".format(str(j*20)))
    file.write("\\\ \hline\n")
    for func in func_list:
        row = func + " & " + "${} \pm {} $".format(GD[func][0],GD[func][1])
        for i in range(len(data_list)):
            row +=  " & " + "${} \pm {} $".format(data_list[i][func][0], data_list[i][func][1])
        row += "\\\ \n"
        file.write(row)
    file.write("\hline\n")
    file.write("\end{tabular}}\n")
    file.write("\end{table}\n")