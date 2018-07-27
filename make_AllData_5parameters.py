import numpy as np
import matplotlib.pyplot as plt

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
    print("routine:",routine_list[ii][1])
    y_true_list = []
    for j in range(1,len(time_table)):
        line1 = time_table[j]
        line1 = line1.split()
        y_true_list.append(float(line1[routine_list[ii][0]]))
    print("len(y_true_list):",len(y_true_list))
    print("y_true_list:",y_true_list)
    
    f = open("./%s/trace.txt" % routine_list[ii][1],"r")
    lines = f.readlines()
    f.close()

    parameter = lines[0]
    parameterlist = parameter.split()

    print(lines[0])
    row_of_T = []
    for j in range(len(parameterlist)):
        if parameterlist[j] != "#j" and parameterlist[j] != "c1" and parameterlist[j] != "c2" and parameterlist[j] != "c3" and parameterlist[j] != "c4" and parameterlist[j] != "c5":
            row_of_T.append( [ j, float(parameterlist[j]) ] )
    
    print("row_of_T:",row_of_T)
    print("len(row_of_T):",len(row_of_T))
    
    graph_data = []
    node_list = []
    median_list = []
    top_40percent_list = []
    bottom_40percent_list = []
    top_95percent_list = []
    bottom_95percent_list = []
    for i in range(len(row_of_T)):
        print("iteration:",i)
        row = row_of_T[i][0]
        print("row:",row)
        T_list = []
        print(len(lines))
        for j in range(1,len(lines)):
            #print("line")
            #print(lines[j])
            #print("j:",j)
            line = lines[j]
            line = line.split()
            #print("line[j]:",line)
            T_list.append(float(line[row]))
        
        sort = np.argsort(T_list)
        #print("i:",i)
        #print("T_list[0]:",T_list[0])
        #print("T_list[1]:",T_list[1])
        #print("T_list[2]:",T_list[2])
        #print("T_list[-1]:",T_list[-1])
        #print("len(T_list):",len(T_list))
        #print("sort:",sort)
        #print("len(sort):",len(sort))
        #print("6th biggest T:",T_list[sort[-6]])
        sorted_T_list = np.sort(T_list)
        print("sorted_T_list;",sorted_T_list)
        median = T_list[sort[2500-1]]
        top_40percent = T_list[sort[3500-1]]
        bottom_40percent = T_list[sort[1500-1]]
        top_95percent = T_list[sort[4875-1]]
        bottom_95percent = T_list[sort[125-1]]
        #graph_data.append( [ row_of_T[i][1], median, top_40percent, bottom_40percent, top_95percent, bottom_95percent ] )
        node_list.append(float(row_of_T[i][1]))
        median_list.append(float(median))
        top_40percent_list.append(float(top_40percent))
        bottom_40percent_list.append(float(bottom_40percent))
        top_95percent_list.append(float(top_95percent))
        bottom_95percent_list.append(float(bottom_95percent))
    data_set.append([routine_list[ii][1],y_true_list,median_list,bottom_95percent_list,top_95percent_list])

print("######################################")
print("#Added All Process#")
f = open("Added_All_trace.txt","r")
lines = f.readlines()
f.close()

parameter = lines[0]
parameterlist = parameter.split()

print(lines[0])
row_of_T = []
for j in range(len(parameterlist)):
    if parameterlist[j] != "#j" and parameterlist[j] != "c1" and parameterlist[j] != "c2" and parameterlist[j] != "c3" and parameterlist[j] != "c4" and parameterlist[j] != "c5":
        row_of_T.append( [ j, float(parameterlist[j]) ] )

print("row_of_T:",row_of_T)
print("len(row_of_T):",len(row_of_T))
    
graph_data = []
node_list = []
median_list = []
top_40percent_list = []
bottom_40percent_list = []
top_95percent_list = []
bottom_95percent_list = []
for i in range(len(row_of_T)):
    print("iteration:",i)
    row = row_of_T[i][0]
    print("row:",row)
    T_list = []
    print(len(lines))
    for j in range(1,len(lines)):
        #print("line")
        #print(lines[j])
        #print("j:",j)
        line = lines[j]
        line = line.split()
            #print("line[j]:",line)
        T_list.append(float(line[row]))
        
    sort = np.argsort(T_list)
        #print("i:",i)
        #print("T_list[0]:",T_list[0])
        #print("T_list[1]:",T_list[1])
        #print("T_list[2]:",T_list[2])
        #print("T_list[-1]:",T_list[-1])
        #print("len(T_list):",len(T_list))
        #print("sort:",sort)
        #print("len(sort):",len(sort))
        #print("6th biggest T:",T_list[sort[-6]])
    sorted_T_list = np.sort(T_list)
    print("sorted_T_list;",sorted_T_list)
    median = T_list[sort[2500-1]]
    top_40percent = T_list[sort[3500-1]]
    bottom_40percent = T_list[sort[1500-1]]
    top_95percent = T_list[sort[4875-1]]
    bottom_95percent = T_list[sort[125-1]]
    #graph_data.append( [ row_of_T[i][1], median, top_40percent, bottom_40percent, top_95percent, bottom_95percent ] )
    node_list.append(float(row_of_T[i][1]))
    median_list.append(float(median))
    top_40percent_list.append(float(top_40percent))
    bottom_40percent_list.append(float(bottom_40percent))
    top_95percent_list.append(float(top_95percent))
    bottom_95percent_list.append(float(bottom_95percent))

mode = "A"
color_list = ["orangered","gold","green","blue","magenta","maroon","gray"]
if mode == "A":
    plt.plot(node_list,median_list, ls='-', lw=1, label='Predict(added ALL)', color="red", marker='o')
    plt.fill_between(node_list, bottom_95percent_list, top_95percent_list, color='red', alpha=0.3, label='95% HPD interval(added ALL)')
    for i in range(len(data_set)):
        plt.plot(node_list, data_set[i][1], ls='-', lw=1, label='True(%s)'%data_set[i][0], color=color_list[i], marker='x')
        plt.plot(node_list, data_set[i][2], ls='-', lw=1, label='Predict(%s)'%data_set[i][0], color=color_list[i], marker='o')
        plt.fill_between(node_list, data_set[i][3], data_set[i][4], color=color_list[i], alpha=0.3, label='95%% HPD interval(%s)'%data_set[i][0])
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
mode = "B"
if mode == "B":
    plt.plot(node_list,median_list, ls='-', lw=1, label='Predict(added ALL)', color="red", marker='o')
    for i in range(len(node_list)):
        plt.plot([node_list[i],node_list[i]] ,[bottom_95percent_list[i], top_95percent_list[i]], color='red', lw=5, alpha=0.5)
    plt.plot([node_list[i],node_list[i]] ,[bottom_95percent_list[i], top_95percent_list[i]], color='red', lw=5, alpha=0.5, label='95% HPD interval(added ALL)')
    for i in range(len(data_set)):
        plt.plot(node_list, data_set[i][1], ls='-', lw=1, label='True(%s)'%data_set[i][0], color=color_list[i], marker='x')
        plt.plot(node_list, data_set[i][2], ls='-', lw=1, label='Predict(%s)'%data_set[i][0], color=color_list[i], marker='o')
        for j in range(len(node_list)):
            plt.plot([node_list[j],node_list[j]], [data_set[i][3][j], data_set[i][4][j]], color=color_list[i], lw=5, alpha=0.5)
        plt.plot([node_list[j],node_list[j]], [data_set[i][3][j], data_set[i][4][j]], color=color_list[i], lw=5, alpha=0.5, label='95%% HPD interval(%s)'%data_set[i][0])
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

f3 = open("AllData.txt","w")
f3.write("#node ")
for i in range(len(routine_list)):
    f3.write("True(%s) Predict(%s) bottom95%%(%s) top95%%(%s)"%(routine_list[i][1],routine_list[i][1],routine_list[i][1],routine_list[i][1]))
    if i != len(routine_list)-1:
        f3.write(" ")
    else:
        f3.write(" Predict(added_ALL) bottom95%(added_ALL) top95%(added_ALL)")
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
            f3.write(str(bottom_95percent_list[i]))
            f3.write(" ")
            f3.write(str(top_95percent_list[i]))
    f3.write("\n")
f3.close()

#    f2 = open("./%s/graph_data.txt" % routine_list[ii][1],"w")
#    f2.write("#P(node), y_true, median, 40percent_top, 40percent_bottom, 95percent_top, 95percent_bottom")
#    f2.write("\n")
#    #for i in range(len(graph_data)):
#    #    for j in range(len(graph_data[i])):
#    #        f2.write(str(graph_data[i][j]))
#    #        if j != len(graph_data[i])-1:
#    #            f2.write(" ")
#    #    f2.write("\n")
#for i in range(len(row_of_T)):
#        f2.write(str(node_list[i]))
#        f2.write(" ")
#        f2.write(str(y_true_list[i]))
#        f2.write(" ")
#        f2.write(str(median_list[i]))
#        f2.write(" ")
#        f2.write(str(top_40percent_list[i]))
#        f2.write(" ")
#        f2.write(str(bottom_40percent_list[i]))
#        f2.write(" ")
#        f2.write(str(top_95percent_list[i]))
#        f2.write(" ")
#        f2.write(str(bottom_95percent_list[i]))
#        f2.write("\n")
#    f2.close()
#
#    print(node_list)
#    plt.plot(node_list, y_true_list, ls='-', lw=1, label='True (Elapse Time)', color="blue", marker='o')
 #   plt.plot(node_list, median_list, ls='-', lw=1, label='Predict (median of Elapse Time)', color="red", marker='o')
 #   plt.fill_between(node_list, bottom_40percent_list, top_40percent_list, color='red', alpha=0.5, label='40% HPD interval of Elapse Time')
 #   plt.fill_between(node_list, bottom_95percent_list, top_95percent_list, color='salmon', alpha=0.5, label='95% HPD interval of Elapse Time')
 #   plt.xlim(node_list[0],node_list[-1])
 #   plt.legend()
 #   plt.xscale("log",basex=2)
 #   plt.yscale("log",basey=10)
 #   plt.xlabel("Number of CPU (P)")
 #   plt.ylabel("Elapse Time [sec]")
 #   plt.grid(which='major', linestyle='-')
 #   plt.grid(which='minor', linestyle=':')
 #   print("len(routine_list):",len(routine_list))
 #   print("ii:",ii)
 #   plt.title("%s" % routine_list[ii][1])
 #   plt.savefig("./%s/graph_forty.png" % routine_list[ii][1])
 #   plt.clf()



