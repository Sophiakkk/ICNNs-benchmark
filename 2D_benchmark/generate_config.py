# # Single global optimizer cases
func_list=["ackley","bukin","dropwave","eggholder","griewank","langermann",
           "levy","levy13","rastrigin","schaffer2","schwefel","shubert",
           "tray","holdertable","schaffer4"]

method_list=["ICNNs"]
total_num = len(func_list)*len(method_list)
T_list = [100]
lr_list = [1e-3,1e-4,1e-5,1e-6,1e-7]
id = 1

with open("benchmark_config.txt","w") as f:
    f.write("ArrayTaskID"+" "+"method"+" "+"func"+" "+"lr"+" "+"T")
    for method in method_list:
        for func in func_list:
            for T in T_list:
                for lr in lr_list:
                    f.write("\n"+str(id)+" "+method+" "+func+" "+str(lr)+" "+str(T))
                    id+=1