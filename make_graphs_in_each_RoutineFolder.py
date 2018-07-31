#####パラメータ設定欄#####
percentage = 40#信頼区間のパーセンテージ
iter=100000
burn=50000
thin=10
##########################

import numpy as np
import matplotlib.pyplot as plt

proportion = percentage/100.0
points = round((iter - burn)/thin)
top_p = 0.5 + proportion/2.0
bottom_p = 0.5 - proportion/2.0

f = open("elapse_time_table.txt","r")
time_table = f.readlines()
f.close()

time_table.pop(0)
item_list = time_table[0]
item_list = item_list[1:].split()
routine_list = []
for i in range(len(item_list)):
    if item_list[i] != "node" and item_list[i] != "SEP" and item_list[i] != "Reducer":
        routine_list.append( [ i, item_list[i] ] )

for ii in range(len(routine_list)):
    y_true_list = []
    for j in range(1,len(time_table)):
        line1 = time_table[j]
        line1 = line1.split()
        y_true_list.append(float(line1[routine_list[ii][0]]))
    
    f = open("./%s/trace.txt" % routine_list[ii][1],"r")
    lines = f.readlines()
    f.close()

    parameter = lines[0]
    parameterlist = parameter.split()

    row_of_T = []
    for j in range(len(parameterlist)):
        if parameterlist[j] != "#j" and parameterlist[j] != "c1" and parameterlist[j] != "c2" and parameterlist[j] != "c3" and parameterlist[j] != "c4" and parameterlist[j] != "c5":
            row_of_T.append( [ j, float(parameterlist[j]) ] )
    
    graph_data = []
    node_list = []
    median_list = []
    top_XXpercent_list = []
    bottom_XXpercent_list = []
    top_95percent_list = []
    bottom_95percent_list = []
    for i in range(len(row_of_T)):
        row = row_of_T[i][0]
        T_list = []
        for j in range(1,len(lines)):
            line = lines[j]
            line = line.split()
            T_list.append(float(line[row]))
        
        sort = np.argsort(T_list)
        sorted_T_list = np.sort(T_list)
        median = T_list[sort[round(points/2) - 1]]
        top_XXpercent = T_list[sort[round(points*top_p) -1]]
        bottom_XXpercent = T_list[sort[round(points*bottom_p) - 1]]
        top_95percent = T_list[sort[round(points*0.975) - 1]]
        bottom_95percent = T_list[sort[round(points*0.025)-1]]
        node_list.append(float(row_of_T[i][1]))
        median_list.append(float(median))
        top_XXpercent_list.append(float(top_XXpercent))
        bottom_XXpercent_list.append(float(bottom_XXpercent))
        top_95percent_list.append(float(top_95percent))
        bottom_95percent_list.append(float(bottom_95percent))
        

    f2 = open("./%s/graph_data.txt" % routine_list[ii][1],"w")
    f2.write("#P(node), y_true, median, %dpercent_top, %dpercent_bottom, 95percent_top, 95percent_bottom"%(percentage,percentage))
    f2.write("\n")
    for i in range(len(row_of_T)):
        f2.write(str(node_list[i]))
        f2.write(" ")
        f2.write(str(y_true_list[i]))
        f2.write(" ")
        f2.write(str(median_list[i]))
        f2.write(" ")
        f2.write(str(top_XXpercent_list[i]))
        f2.write(" ")
        f2.write(str(bottom_XXpercent_list[i]))
        f2.write(" ")
        f2.write(str(top_95percent_list[i]))
        f2.write(" ")
        f2.write(str(bottom_95percent_list[i]))
        f2.write("\n")
    f2.close()

    plt.plot(node_list, y_true_list, ls='-', lw=1, label='True (Elapse Time)', color="blue", marker='o')
    plt.plot(node_list, median_list, ls='-', lw=1, label='Predict (median of Elapse Time)', color="red", marker='o')
    plt.fill_between(node_list, bottom_XXpercent_list, top_XXpercent_list, color='red', alpha=0.5, label='%d%% HPD interval of Elapse Time'%percentage)
    plt.fill_between(node_list, bottom_95percent_list, top_95percent_list, color='salmon', alpha=0.5, label='95% HPD interval of Elapse Time')
    plt.xlim(node_list[0],node_list[-1])
    plt.legend()
    plt.xscale("log",basex=2)
    plt.yscale("log",basey=10)
    plt.xlabel("Number of CPU (P)")
    plt.ylabel("Elapse Time [sec]")
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle=':')
    plt.title("%s" % routine_list[ii][1])
    plt.savefig("./%s/graph_data.png" % routine_list[ii][1])
    plt.clf()
