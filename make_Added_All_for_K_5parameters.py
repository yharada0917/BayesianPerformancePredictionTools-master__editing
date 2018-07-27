#import matplotlib.pyplot as plt

routine_list = ["pdormtr","pdstedc","pdsytrd","pdsygst","pdpotrf","rest"]

trace_data_set = []
for routine in routine_list:
    f = open("./%s/trace.txt"%routine,"r")
    lines = f.readlines()
    f.close()

    trace_data =  []
    for i in range(len(lines)):
        if i != 0:
            line = lines[i]
            line = line.split()
            trace_data.append(line)
    trace_data_set.append(trace_data)

print("len(trace_data_set)=6:",len(trace_data_set))
for i in range(6):
    print("len(trace_data_set[%d]):"%i,len(trace_data_set[i]))
print("len(trace_data_set[0][0][0]):",len(trace_data_set[0][0][0]))
#print("len(trace_data_set[0][0][1]):",len(trace_data_set[0][0][1]))

Added_All_data = []
for i in range(5000):
    line2 = []
    for j in range(13):
        if j == 0:
            line2.append(str(i+1))
        else:
            total = 0.0
            for k in range(len(routine_list)):
                total = total+float(trace_data_set[k][i][j])
            line2.append(str(total))
    Added_All_data.append(line2)

print("len(Added_All_data)=5000:",len(Added_All_data))
for i in range(5000):
    print("len(Added_All_data)[%d]=13"%i,len(Added_All_data[i]))


f = open("Added_All_trace.txt","w")
f.write("#j c1 c2 c3 c4 c5 4 16 64 256 1024 4096 10000")
for i in range(len(Added_All_data)):
    f.write("\n")
    line3 = Added_All_data[i]
    for j in range(len(line3)):
        f.write(line3[j])
        if j != len(line3)-1:
            f.write(" ")
f.close()        
