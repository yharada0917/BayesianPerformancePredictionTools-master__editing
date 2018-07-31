#####パラメータ設定欄#####
mode = ["A","B","C"]#エラーバーの描画スタイルを選択（複数選択可）
#"A":塗りつぶし　"B":鉛直方向の太線　"C":点線
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

data_set = []
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
        node_list.append(float(row_of_T[i][1]))
        median_list.append(float(median))
        top_XXpercent_list.append(float(top_XXpercent))
        bottom_XXpercent_list.append(float(bottom_XXpercent))
    data_set.append([routine_list[ii][1],y_true_list,median_list,bottom_XXpercent_list,top_XXpercent_list])

f = open("Added_All_trace.txt","r")
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
for i in range(len(row_of_T)):
    row = row_of_T[i][0]
    T_list = []
    for j in range(1,len(lines)):
        line = lines[j]
        line = line.split()
        T_list.append(float(line[row]))
        
    sort = np.argsort(T_list)
    sorted_T_list = np.sort(T_list)
    median = T_list[sort[round(points/2 - 1)]]
    top_XXpercent = T_list[sort[round(points*top_p) -1]]
    bottom_XXpercent = T_list[sort[round(points*bottom_p) - 1]]
    node_list.append(float(row_of_T[i][1]))
    median_list.append(float(median))
    top_XXpercent_list.append(float(top_XXpercent))
    bottom_XXpercent_list.append(float(bottom_XXpercent))

color_list = ["orangered","gold","green","blue","magenta","maroon","gray"]
if "A" in mode:
    plt.plot(node_list,median_list, ls='-', lw=1, label='Predict(added ALL)', color="red", marker='o')
    plt.fill_between(node_list, bottom_XXpercent_list, top_XXpercent_list, color='red', alpha=0.3, label='%d%% HPD interval(added ALL)'%percentage)
    for i in range(len(data_set)):
        plt.plot(node_list, data_set[i][1], ls='-', lw=1, label='True(%s)'%data_set[i][0], color=color_list[i], marker='x')
        plt.plot(node_list, data_set[i][2], ls='-', lw=1, label='Predict(%s)'%data_set[i][0], color=color_list[i], marker='o')
        plt.fill_between(node_list, data_set[i][3], data_set[i][4], color=color_list[i], alpha=0.3, label='%d%% HPD interval(%s)'%(percentage,data_set[i][0]))
    plt.xlim(node_list[0],node_list[-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xscale("log",basex=2)
    plt.yscale("log",basey=10)
    plt.xlabel("Number of CPU (P)")
    plt.ylabel("Elapse Time [sec]")
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle=':')
    plt.title("Predicting")
    plt.savefig("AllData_fill.png",dpi=300,bbox_inches='tight')
    plt.clf()
if "B" in mode:
    plt.plot(node_list,median_list, ls='-', lw=1, label='Predict(added ALL)', color="red", marker='o')
    for i in range(len(node_list)):
        plt.plot([node_list[i],node_list[i]] ,[bottom_XXpercent_list[i], top_XXpercent_list[i]], color='red', lw=5, alpha=0.5)
    plt.plot([node_list[i],node_list[i]] ,[bottom_XXpercent_list[i], top_XXpercent_list[i]], color='red', lw=5, alpha=0.5, label='%d%% HPD interval(added ALL)'%percentage)
    for i in range(len(data_set)):
        plt.plot(node_list, data_set[i][1], ls='-', lw=1, label='True(%s)'%data_set[i][0], color=color_list[i], marker='x')
        plt.plot(node_list, data_set[i][2], ls='-', lw=1, label='Predict(%s)'%data_set[i][0], color=color_list[i], marker='o')
        for j in range(len(node_list)):
            plt.plot([node_list[j],node_list[j]], [data_set[i][3][j], data_set[i][4][j]], color=color_list[i], lw=5, alpha=0.5)
        plt.plot([node_list[j],node_list[j]], [data_set[i][3][j], data_set[i][4][j]], color=color_list[i], lw=5, alpha=0.5, label='%d%% HPD interval(%s)'%(percentage,data_set[i][0]))
    plt.xlim(node_list[0],node_list[-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xscale("log",basex=2)
    plt.yscale("log",basey=10)
    plt.xlabel("Number of CPU (P)")
    plt.ylabel("Elapse Time [sec]")
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle=':')
    plt.title("Predicting")
    plt.savefig("AllData_bar.png",dpi=300,bbox_inches='tight')
    plt.clf()
if "C" in mode:
    plt.plot(node_list,median_list, ls='-', lw=1, label='Predict(added ALL)', color="red", marker='o')
    plt.plot(node_list,bottom_XXpercent_list, ls='--', lw=1, label='%d%% HPD interval(added ALL)'%percentage, color="red", marker='o')
    plt.plot(node_list,bottom_XXpercent_list, ls='--', lw=1, color="red", marker='o')
    for i in range(len(data_set)):
        plt.plot(node_list, data_set[i][1], ls='-', lw=1, label='True(%s)'%data_set[i][0], color=color_list[i], marker='x')
        plt.plot(node_list, data_set[i][2], ls='-', lw=1, label='Predict(%s)'%data_set[i][0], color=color_list[i], marker='o')
        plt.plot(node_list, data_set[i][3], ls='--', lw=1, label='%d%% HPD interval(%s)'%(percentage,data_set[i][0]), color=color_list[i], marker='o')
        plt.plot(node_list, data_set[i][3], ls='--', lw=1, color=color_list[i], marker='o')
    plt.xlim(node_list[0],node_list[-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xscale("log",basex=2)
    plt.yscale("log",basey=10)
    plt.xlabel("Number of CPU (P)")
    plt.ylabel("Elapse Time [sec]")
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle=':')
    plt.title("Predicting")
    plt.savefig("AllData_dotted.png",dpi=300,bbox_inches='tight')
    plt.clf()

f3 = open("AllData.txt","w")
f3.write("#node ")
for i in range(len(routine_list)):
    f3.write("True(%s) Predict(%s) bottom%d%%(%s) top%d%%(%s)"%(routine_list[i][1],routine_list[i][1],percentage,routine_list[i][1],percentage,routine_list[i][1]))
    if i != len(routine_list)-1:
        f3.write(" ")
    else:
        f3.write(" Predict(added_ALL) bottom%d%%(added_ALL) top%d%%(added_ALL)"%(percentage,percentage))
f3.write("\n")
for i in range(len(node_list)):
    f3.write(str(node_list[i]))
    f3.write(" ")
    for j in range(len(data_set)):
        f3.write(str(data_set[j][1][i]))
        f3.write(" ")
        f3.write(str(data_set[j][2][i]))
        f3.write(" ")
        f3.write(str(data_set[j][3][i]))
        f3.write(" ")
        f3.write(str(data_set[j][4][i]))
        f3.write(" ")
        if j == len(data_set)-1:
            f3.write(str(median_list[i]))
            f3.write(" ")
            f3.write(str(bottom_XXpercent_list[i]))
            f3.write(" ")
            f3.write(str(top_XXpercent_list[i]))
    f3.write("\n")
f3.close()
