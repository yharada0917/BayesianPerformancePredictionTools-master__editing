# -*- coding: utf-8 -*-
import pymc
import numpy as np
import matplotlib.pyplot as plt
import copy, math, os, sys, collections, random, argparse

def main(l, y_range):
    f = open('elapse_time_table.txt', 'r')
    dict_value = collections.OrderedDict()
    cnt = 0
    table_item = ''

    for line in (f):
        if line[0] == '#' and not cnt == 0:
            table_item = line[1:].split()

            for nd, dic in enumerate(table_item):
                dict_value.update({dic:[]})
            cnt += 1

        if not line[0] == '#':
            data = line.split()
            for n, item in enumerate(table_item):
                tmp_list = dict_value[item]
                tmp_list.append(float(data[n]))
                dict_value.update({item:tmp_list})

        cnt += 1

    routine_list = []
    for item in table_item:
        if not item in ['node', 'SEP', 'Reducer']:
            routine_list.append(item)

    parameter = ['c1','c2','c3','c4','c5','eps','tau']

    def model(x, y):
        c1 = pymc.Uniform('c1', lower=0, upper=100000)
        c2 = pymc.Uniform('c2', lower=0, upper=100000)
        c3 = pymc.Uniform('c3', lower=0, upper=100000)
        c4 = pymc.Uniform('c4', lower=0, upper=10000000)
        c5 = pymc.Uniform('c5', lower=0, upper=100000)
        eps = pymc.Uniform('eps', lower=0, upper=0.5)

        @pymc.deterministic
        def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
            if l:
                x_list = []
                for i in range(len(x)):
                    if x[i]>100:
                        term5 = 0.0
                    else:
                        term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
                    x_list.append(np.log((c4 / np.exp(2.0*x[i])) + (c1 / np.exp(x[i])) + c2 + (c3 * x[i]) + term5))
                return x_list
            else:
                return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))


        @pymc.deterministic
        def tau(eps=eps):
            return np.power(eps, -2)

        y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
        return locals()

    split_start = 0
    n_split = 3#3

    x_true = copy.deepcopy(dict_value['node'][split_start:])
    node = np.log(dict_value['node'][split_start:n_split])
    #print("node:",node)

    print(dict_value['node'][split_start:n_split])
    data_list = []
    print("routine_list:",routine_list)
    for i in routine_list:
        print(i, dict_value[i][split_start:n_split])
        y_true = copy.deepcopy(dict_value[i][split_start:])
        time = copy.deepcopy(dict_value[i][split_start:n_split])
        time = np.log(time)

        pymc.numpy.random.seed(0)
        mcmc = pymc.MCMC(model(node, time))
        mcmc.sample(iter=100000, burn=50000, thin=10)

        os.system('mkdir %s' % i)

        trace_list = []
        for j in parameter:
            pymc.Matplot.plot(mcmc.trace(j))
            mcmctrace = np.array(mcmc.trace("%s" % j, chain=None)[:])
            print("mcmctrace:",mcmctrace)
            trace_list.append(mcmctrace)
            pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
            plt.clf()
            plt.close()

        print("len(trace_list)=5:",len(trace_list))
        print("len(trace_list[0])=5000:",len(trace_list[0]))
        print("trace_list[0]:",trace_list[0])
        j_list = [i+1 for i in range(len(trace_list[0]))]
        T_list = []
        for P in x_true:
            T_list_single = []
            for C in range(len(trace_list[0])):
                c1 = trace_list[0][C]
                c2 = trace_list[1][C]
                c3 = trace_list[2][C]
                c4 = trace_list[3][C]
                c5 = trace_list[4][C]
                if l:
                    if P>100:
                        term5 = 0.0
                    else:
                        term5 = (c5*P)/(np.exp(0.5*P))
                    T_list_single.append((c4 / P**2.0) + (c1 / P) + c2 + (c3 * np.log(P)) + term5)
                else:
                    T_list_single.append((c4 / P**2.0) + (c1 / P) + c2 + (c3 * P) + c5*(np.log(P)/np.sqrt(P)))
            T_list.append(T_list_single)

        print("len(trace_list[0])",len(trace_list[0]))
        print("trace_list[0]")
        print(trace_list[0])
        f1 = open('./%s/trace.txt' % i,"w")
        f1.write("#j c1 c2 c3 c4 c5")
        for P in x_true:
            f1.write(" %s"% int(P))
        f1.write("\n")
        for C in range(len(trace_list[0])):
            f1.write(str(j_list[C]))
            f1.write(" ")
            f1.write(str(trace_list[0][C]))
            f1.write(" ")
            f1.write(str(trace_list[1][C]))
            f1.write(" ")
            f1.write(str(trace_list[2][C]))
            f1.write(" ")
            f1.write(str(trace_list[3][C]))
            f1.write(" ")
            f1.write(str(trace_list[4][C]))
            for CC in range(len(T_list)):
                f1.write(" ")
                f1.write(str(T_list[CC][C]))
            f1.write("\n")
        f1.close()

        sys.stdout = open('./%s/detail.txt' % i, 'w')
        mcmc.summary()
        sys.stdout.close()
        sys.stdout = sys.__stdout__

        c1_median = np.median(mcmc.trace('c1', chain=None)[:])
        c2_median = np.median(mcmc.trace('c2', chain=None)[:])
        c3_median = np.median(mcmc.trace('c3', chain=None)[:])
        c4_median = np.median(mcmc.trace('c4', chain=None)[:])
        c5_median = np.median(mcmc.trace('c5', chain=None)[:])

        c1_min = mcmc.stats()['c1']['95% HPD interval'][0]
        c1_max = mcmc.stats()['c1']['95% HPD interval'][1]
        c2_min = mcmc.stats()['c2']['95% HPD interval'][0]
        c2_max = mcmc.stats()['c2']['95% HPD interval'][1]
        c3_min = mcmc.stats()['c3']['95% HPD interval'][0]
        c3_max = mcmc.stats()['c3']['95% HPD interval'][1]
        c4_min = mcmc.stats()['c4']['95% HPD interval'][0]
        c4_max = mcmc.stats()['c4']['95% HPD interval'][1]
        c5_min = mcmc.stats()['c5']['95% HPD interval'][0]
        c5_max = mcmc.stats()['c5']['95% HPD interval'][1]
        y_pre = []
        y_pre_min = []
        y_pre_max = []

        for P in x_true:
            if l:
                if P>100:
                    term5_median = 0.0
                    term5_min = 0.0
                    term5_max = 0.0
                else:
                    term5_median = (c5_median*P)/(np.exp(0.5*P))
                    term5_min = (c5_min*P)/(np.exp(0.5*P))
                    term5_max = (c5_max*P)/(np.exp(0.5*P))
                y_pre.append((c4_median / P**2.0) + (c1_median / P) + c2_median + (c3_median * np.log(P)) + term5_median)
                y_pre_min.append((c4_min / P**2.0) + (c1_min / P) + c2_min + (c3_min * np.log(P)) + term5_min)
                y_pre_max.append((c4_max / P**2.0) + (c1_max / P) + c2_max + (c3_max * np.log(P)) + term5_max)
            else:
                y_pre.append((c4_median / P**2.0) + (c1_median / P) + c2_median + (c3_median * P) + c5_median*(np.log(P)/np.sqrt(P)))
                y_pre_min.append((c4_min / P**2.0) + (c1_min / P) + c2_min + (c3_min * P) + c5_min*(np.log(P)/np.sqrt(P)))
                y_pre_max.append((c4_max / P**2.0) + (c1_max / P) + c2_max + (c3_max * P) + c5_max*(np.log(P)/np.sqrt(P)))

        data_list.append([y_true, y_pre, i])
        plt.plot(x_true, y_true, ls='-', lw=1, label='True', marker='o')
        plt.plot(x_true, y_pre, label='Predict (median of c1, c2, c3, c4, c5)', marker='o')
        plt.fill_between(x_true, y_pre_min, y_pre_max, color='r', alpha=0.1, label='95% HPD interval of c1, c2, c3, c4, c5')
        plotting(plt, './%s/graph.png' % i, i, '(%s)' % i, x_true, [min(y_pre_min), max(y_pre_max)])
        f2 = open('./%s/text.txt' % i,'w')
        f2.write("#x_true y_true y_pre y_pre_min y_pre_max")
        f2.write("\n")
        for k in range(len(x_true)):
            f2.write(str(x_true[k]))
            f2.write(" ")
            f2.write(str(y_true[k]))
            f2.write(" ")
            f2.write(str(y_pre[k]))
            f2.write(" ")
            f2.write(str(y_pre_min[k]))
            f2.write(" ")
            f2.write(str(y_pre_max[k]))
            f2.write("\n")
        f2.close()

    cmap = plt.get_cmap("tab10")
    #plt.plot(x_true, data_list[0][0], label='True (total)', marker='o', color=cmap(0))
    #plt.plot(x_true, data_list[0][1], label='Predict (total)', marker='o', color=cmap(0), linestyle='--')
    #plt.plot(x_true, data_list[3][0], label='True (pdsytrd)', marker='o', color=cmap(1))
    #plt.plot(x_true, data_list[3][1], label='Predict (pdsytrd)', marker='o', color=cmap(1), linestyle='--')
    #plt.plot(x_true, data_list[4][0], label='True (pdsygst)', marker='o', color=cmap(2))
    #plt.plot(x_true, data_list[4][1], label='Predict (pdsygst)', marker='o', color=cmap(2), linestyle='--')
    for jjj in range(len(data_list)):
        if data_list[jjj][2] == "Total":
            plt.plot(x_true, data_list[jjj][0], label='True (total)', marker='o', color=cmap(0))
            plt.plot(x_true, data_list[jjj][1], label='Predict (total)', marker='o', color=cmap(0), linestyle='--')
        if data_list[jjj][2] == "pdsytrd":
            plt.plot(x_true, data_list[jjj][0], label='True (pdsytrd)', marker='o', color=cmap(1))
            plt.plot(x_true, data_list[jjj][1], label='Predict (pdsytrd)', marker='o', color=cmap(1), linestyle='--')
        if data_list[jjj][2] == "pdsygst":
            plt.plot(x_true, data_list[jjj][0], label='True (pdsygst)', marker='o', color=cmap(2))
            plt.plot(x_true, data_list[jjj][1], label='Predict (pdsygst)', marker='o', color=cmap(2), linestyle='--')

    #combined = [a + b + c + d + e + f for (a, b, c, d, e, f) in zip(data_list[1][1], data_list[2][1], data_list[3][1], data_list[4][1], data_list[5][1], data_list[6][1])]
    combined = [0.0 for i in range(len(x_true))]
    for kkk in range(len(data_list)):
        if not data_list[kkk][2] == "Total":
                for ll in range(len(combined)):
                    combined[ll] += data_list[kkk][1][ll]
    plt.plot(x_true, combined, label='Predict(added ALL)', marker='o', color=cmap(4), linestyle='--')
    plotting(plt, './graph_predict.png', 'Predicting Total, pdsytrd and pdsygst', '', x_true, y_range)
    f3 = open('./text_predict.txt',"w")
    f3.write("#x_true combined(added ALL)")
    f3.write("\n")
    for kk in range(len(x_true)):
        f3.write(str(x_true[kk]))
        f3.write(" ")
        f3.write(str(combined[kk]))
        f3.write("\n")
    f3.close()


def plotting(plt, path, title, y_label, node, y_range):
    plt.xlim(node[0], node[-1])
    plt.xticks(node)

    if not y_range[1] == 'default':
        plt.ylim(y_range[0], y_range[1])

    plt.legend()
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)

    plt.xlabel("Number of CPU (P)")
    plt.ylabel('Elapse Time %s [sec]' % y_label)

    plt.title('%s' % title)
    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle=':')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(right=0.5)
    plt.savefig('%s' % path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-yr', '--y_range',
                        nargs = 2,
                        type = float,
                        default = [10, 'default'],
                        help = 'y range (min max)')

    parser.add_argument('-l', '--log',
                        action = 'store_true',
                        help = 'log flag (default = False)')

    args = parser.parse_args()
    y_range = args.y_range
    l = args.log
    main(l, y_range)
