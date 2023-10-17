headers = ["ackley","bukin","tray","dropwave","eggholder","griewank","holdertable",
             "levy","levy13","rastrigin","schaffer2","schaffer4","schwefel", "shubert"]

with open ("summary_2d_T50.txt", "r") as f:
    data50 = {}
    for line in f:
        if line.startswith("Senario"):
            func = line.split()[4]
            data50[func] = []
        if line.startswith("avg error (output)"):
            data50[func].append(line.split()[3])
        if line.startswith("std error (output)"):
            data50[func].append(line.split()[3])

with open ("summary_2d_T100.txt", "r") as f:
    data100 = {}
    for line in f:
        if line.startswith("Senario"):
            func = line.split()[4]
            data100[func] = []
        if line.startswith("avg error (output)"):
            data100[func].append(line.split()[3])
        if line.startswith("std error (output)"):
            data100[func].append(line.split()[3])

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

with open ("latex_summary.txt", "w") as file:
    file.write("\\begin{table}\n")
    file.write("\\centering\n")
    file.write("\\caption{Benchmark Optimization Errors: this table shows y Euclidien distance}\n")
    file.write("\\label{benchmark_yerrors_icnns}\n")
    file.write("\\resizebox{\\textwidth}{!}{\n")
    file.write("\\begin{tabular}{c|ccc}\n")
    file.write("\hline\n")
    file.write("Test Senerio & GD & ICNNs(T50) & ICNNs(T100) \\\ \hline\n")
    for func in headers:
       file.write(func + " & " + "${} \pm {} $".format(GD[func][0],GD[func][1]) + " & " + "${} \pm {} $".format(data50[func][0], data50[func][1]) + " & " + "${} \pm {} $".format(data100[func][0], data100[func][1]) + "\\\ \n")
    file.write("\hline\n")
    file.write("\end{tabular}}\n")
    file.write("\end{table}\n")