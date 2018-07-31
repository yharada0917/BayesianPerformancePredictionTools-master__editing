#####パラメータ設定欄#####
first_line = "#j c1 c2 c3 c4 c5 4 16 64 256 1024 4096 10000"#trace.txtの1行目※パラメータ数やルーチン（京かOFPか）などによって違う
routine_list = ["pdormtr","pdstedc","pdsytrd","pdsygst","pdpotrf","rest"]
iter=100000
burn=50000
thin=10
##########################

items = len(first_line.split())
routines = len(routine_list)
points = round((iter - burn)/thin)


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

Added_All_data = []
for i in range(points):
    line2 = []
    for j in range(items):
        if j == 0:
            line2.append(str(i+1))
        else:
            total = 0.0
            for k in range(routines):
                total = total+float(trace_data_set[k][i][j])
            line2.append(str(total))
    Added_All_data.append(line2)

f = open("Added_All_trace.txt","w")
f.write(first_line)
for i in range(len(Added_All_data)):
    f.write("\n")
    line3 = Added_All_data[i]
    for j in range(len(line3)):
        f.write(line3[j])
        if j != len(line3)-1:
            f.write(" ")
f.close()        
