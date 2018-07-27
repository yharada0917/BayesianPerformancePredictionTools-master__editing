import matplotlib.pyplot as plt

bins = 50

f = open("trace.txt","r")
lines = f.readlines()
f.close()

item_line = lines[0]
items = item_line.split()
parameter_list = []
for i in range(len(items)):
    parameter = items[i]
    if parameter in ["c1","c2","c3","c4","c5"]:
        parameter_list.append([parameter,i])

for i in range(len(parameter_list)):
    parameter_name = parameter_list[i][0]
    row_number = parameter_list[i][1]
    data_list = []
    for j in range(len(lines)):
        if j != 0:
            line = lines[j]
            line = line.split()
            data_list.append(float(line[row_number]))
    max_data = max(data_list)
    plt.hist(data_list,bins=bins)
    plt.title("%s histogram (bins = %d,delta = %f,max = %f)"%(parameter_name,bins,max_data/bins,max_data))
    plt.xlim(0,6000)
    plt.ylim(0,900)
    plt.savefig("%s_histogram.png"%parameter_name)
    plt.clf()


    
    

    
