# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:40:51 2016

Work OK 180821

# Modelを関数化して実行してみる 5 Params
@author: harada

runfile('C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_5parameters_01b.py', 
        wdir='C:\\scripts\\BayesianPerformancePredictionTools-master')
runfile('C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_5PM_parameters_01.py', 
        wdir='C:\\scripts\\BayesianPerformancePredictionTools-master')
runfile('C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_5parameters_13.py', 
        wdir='C:\\scripts\\BayesianPerformancePredictionTools-master')
"""


import pymc
import numpy as np
#import pandas
import matplotlib.pyplot as plt
import copy, math, os, sys, collections, random, argparse



def main_13(l, y_range):
    
    # モジュールのインポート #-----------------------------------------------------
    ## いちおうパスを追加しておく
    #sys.path.append('C:\\scripts')
    #sys.path.append('C:\\scripts\\BayesianPerformancePredictionTools-master')
    
    #from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine
    #from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_df
    from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_03
    #from tools_for_BayesianPerformancePredictionTools import save_txt_f1
    #from tools_for_BayesianPerformancePredictionTools import save_txt_f2
    #from tools_for_BayesianPerformancePredictionTools import save_txt_f3
    from tools_for_BayesianPerformancePredictionTools import values_at
    #from tools_for_BayesianPerformancePredictionTools import plotting_02
    #from tools_for_BayesianPerformancePredictionTools import plotting
    
    from models_for_BayesianPerformancePredictionTools import test_model_153
    from models_for_BayesianPerformancePredictionTools import test_model_251
    # モジュールのインポート END #-----------------------------------------------------
    
    
    
    # 事前準備 #-----------------------------------------------------
    #ofp = 'C:\scripts\BayesianPerformancePredictionTools-master\elapse_time_table.txt' # open file path
    #dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node']) # dict_value, routine_list = get_dict_and_routine(ofp)
    
    
    ofp = 'C:\scripts\BayesianPerformancePredictionTools-master\elapse_time_table_pm_fake.txt' # open file path # elapse_time_table_pm
    dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node', 'msize']) # dict_value, routine_list = get_dict_and_routine(ofp)
    
    
    nodes_in_sampling = [4,16,64] # MCMC教師として使用するNode数 # for dict version
    msize_in_sampling = [10,30,100] # MCMC教師として使用するNode数 # for dict version
    #parameter = ['c1','c2','c3','c4','c5','eps','tau'] # parameter = ['c1','c2','c3','eps','tau']
    parameter = ['c11','c21','c31','c41','c51',
                 'c12','c22','c32','c42','c52','eps','tau']
    header_names = ['A01', 'A02', 'A03', 'A04', 'A05'] # MCMCとして使用するHeader
    #header_names = ['Total', 'pdormtr', 'pdstedc', 'pdsytrd', 'pdsygst', 'pdpotrf', 'rest'] # MCMCとして使用するHeader
    
    
    bm_list_1 = list(map(lambda elem : elem in nodes_in_sampling, dict_value['#node'])) # 条件の一致するものをTrue
    bm_list_2 = list(map(lambda elem : elem in msize_in_sampling, dict_value['msize'])) # 条件の一致するものをTrue
    bm_list = np.logical_and(bm_list_1,bm_list_2) # sampling = np.where(bm_list==True)
    # sampling = [dict_value['#node'].index(ixnom) for ixnom in nodes_in_sampling if ixnom in dict_value['#node'] ] # 教師データとして使うエントリーのインデックスを取ってくる
    header_in_use = [colnom for colnom in header_names if colnom in dict_value.keys()] # 実際にdf.indexにあるものだけに限定
    
    
    xp_sample = np.array(dict_value['#node'])[bm_list].copy()
    xp_sample_l = np.log(xp_sample).copy()
    xp_predict = np.array(dict_value['#node']).copy()

    xm_sample = np.array(dict_value['msize'])[bm_list].copy()
    xm_sample_l = np.log(xm_sample).copy()
    xm_predict = np.array(dict_value['msize']).copy()
    
    #x_sample = copy.deepcopy(values_at(dict_value['#node'], sampling)) # df.loc[sampling, '#node'].copy()
    #x_sample_l = list(map(lambda x:np.log(x), x_sample ))
    #x_predict = copy.deepcopy(dict_value['#node']) 
    # 事前準備 END #-----------------------------------------------------
    
    
    
    
    
    # 本体 #-----------------------------------------------------
    for i in header_in_use : # FOR文
        y_sample = np.array(dict_value[i])[bm_list].copy()
        print(i, y_sample ) # ; print()
        #print(i, values_at(dict_value[i], sampling)) # ; print()
        #y_sample = copy.deepcopy(values_at(dict_value[i], sampling))
        
        mcmc = pymc.MCMC(test_model_251(xp_sample, xm_sample, y_sample))
        mcmc.sample(iter=100000, burn=50000, thin=10)
        
        os.system('mkdir %s' % i) # 出力先DIR作成
        trace_list = []
        for j in parameter:
            pymc.Matplot.plot(mcmc.trace(j))
            mcmctrace = np.array(mcmc.trace("%s" % j, chain=None)[:])
            print("mcmctrace:",mcmctrace)
            trace_list.append(mcmctrace)
            pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
            plt.clf()
            plt.close()
    
    ##for i in header_in_use : # FOR文
    #i = header_in_use[0]
    #y_sample = copy.deepcopy(values_at(dict_value[i], sampling))
    #
    #mcmc = pymc.MCMC(test_model_153(np.array(x_sample_l), y_sample))
    #mcmc.sample(iter=100000, burn=50000, thin=10)
    #
    #trace_list = []
    #for j in parameter:
    #    pymc.Matplot.plot(mcmc.trace(j))
    #    mcmctrace = np.array(mcmc.trace("%s" % j, chain=None)[:])
    #    print("mcmctrace:",mcmctrace)
    #    trace_list.append(mcmctrace)
    #    pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
    #    plt.clf()
    #    plt.close()
    # 本体 END #-----------------------------------------------------
    return 0














#-----------------------------------------------------
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
    main_13(l, y_range)
else:
    print('skip run "__main__"')
#-----------------------------------------------------




# # # ここまでが確認ずみ作業

# # # 以下は下書き







# old #-----------------------------------------------------

def main_03(l, y_range):
    l=True # 引数を渡す　#　Log
    
    
    # load parameters #----------
    ofp = 'C:\scripts\BayesianPerformancePredictionTools-master\elapse_time_table.txt' # open file path
    # parameter = ['c1','c2','c3','eps','tau']
    parameter = ['c1','c2','c3','c4','c5','eps','tau']
    
    #nodes_in_sampling = ['4.0000', '16.000', '64.000'] # MCMC教師として使用するNode数 # for Pandas version
    nodes_in_sampling = [4,16,64] # MCMC教師として使用するNode数 # for dict version
    header_names = ['Total', 'pdormtr', 'pdstedc', 'pdsytrd', 'pdsygst', 'pdpotrf', 'rest'] # MCMCとして使用するHeader
    
    
    
    
    sys.path.append('C:\scripts\BayesianPerformancePredictionTools-master')
    sys.path.append('C:\scripts')
    # load parameters END #----------
    
    
    dict_value, routine_list = get_dict_and_routine(ofp)
    # df = get_dict_and_routine_df(ofp).astype(np.float32)
    
    #sampling = [ixnom for ixnom in nodes_in_sampling if ixnom in df.index] # 実際にdf.indexにあるものだけに限定
    sampling = [dict_value['node'].index(ixnom) for ixnom in nodes_in_sampling if ixnom in dict_value['node'] ] # 実際にdf.indexにあるものだけに限定
    #header_in_use = [colnom for colnom in header_names if colnom in df.columns]
    header_in_use = [colnom for colnom in header_names if colnom in dict_value.keys()]
    
    #x_true = copy.deepcopy(dict_value['node'][split_start : ])
    x_sample = copy.deepcopy(values_at(dict_value['node'], sampling)) # df.loc[sampling, '#node'].copy()
    x_sample_l = list(map(lambda x:np.log(x), x_sample ))
    x_predict = copy.deepcopy(dict_value['node']) 
    
    for i in header_in_use : # FOR文
        # print(i, dict_value[i][split_start:n_split])
        print(i, values_at(dict_value[i], sampling)) # df.loc[sampling, [i]])
        # y_true = copy.deepcopy(dict_value[i][split_start:])
        y_predict = copy.deepcopy(dict_value[i]) # df.loc[:, i].copy()
        # time = copy.deepcopy(dict_value[i][split_start:n_split])
        y_sample = copy.deepcopy(values_at(dict_value[i], sampling) ) # df.loc[sampling, i].copy()
        # time = time
    
        pymc.numpy.random.seed(0)
        mcmc = pymc.MCMC(test_model_151(x_sample_l, y_sample, l))
        mcmc.sample(iter=100000, burn=50000, thin=10)
        #mcmc = pymc.MCMC(model(node, time))
        #mcmc.sample(iter=100000, burn=50000, thin=10)
    
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
    return 0



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
        c1 = pymc.Uniform('c1', lower=0, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
        c2 = pymc.Uniform('c2', lower=0, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
        c3 = pymc.Uniform('c3', lower=0, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
        c4 = pymc.Uniform('c4', lower=0, upper=10000000)#c4の初期分布（lowerからupperまでの一様分布）
        c5 = pymc.Uniform('c5', lower=0, upper=100000)#c5の初期分布（lowerからupperまでの一様分布）
        eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）

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
    n_split = 3#教師データの数（n_split個目までを教師データとして学習）

    x_true = copy.deepcopy(dict_value['node'][split_start:])
    node = np.log(dict_value['node'][split_start:n_split])

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





## Array で渡してみる　これでも問題なし
#def test_model_153(x, y):
#    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
#    c2 = pymc.Uniform('c2', lower=0, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
#    c3 = pymc.Uniform('c3', lower=0, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
#    c4 = pymc.Uniform('c4', lower=0, upper=10000000)#c4の初期分布（lowerからupperまでの一様分布）
#    c5 = pymc.Uniform('c5', lower=0, upper=100000)#c5の初期分布（lowerからupperまでの一様分布）
#    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
#    
#    @pymc.deterministic
#    def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
#        return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))
#    
#    #@pymc.deterministic
#    #def function_l(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l:
#    #    obj = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + (c5 * elem)/(np.exp(0.5*elem))), x ))
#    #    return obj
#    #
#    #@pymc.deterministic
#    #def function_l100(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l: if x[i]>100:
#    #    term5 = 0.0 # term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
#    #    obj = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + term5), x ))
#    #    return obj
#    
#    @pymc.deterministic
#    def tau(eps=eps):
#        return np.power(eps, -2)
#    
#    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
#    return locals()





#def get_dict_and_routine(ofp):
#    # usage ; dict_value, routine_list = get_dict_and_routine(ofp)
#    f = open(ofp, 'r')
#    dict_value = collections.OrderedDict()
#    cnt = 0
#    table_item = ''
#    
#    for line in (f):
#        if line[0] == '#' and not cnt == 0:
#            table_item = line[1:].split()
#    
#            for nd, dic in enumerate(table_item):
#                dict_value.update({dic:[]})
#            cnt += 1
#    
#        if not line[0] == '#':
#            data = line.split()
#            for n, item in enumerate(table_item):
#                tmp_list = dict_value[item]
#                tmp_list.append(float(data[n]))
#                dict_value.update({item:tmp_list})
#        cnt += 1
#    
#    routine_list = []
#    for item in table_item:
#        if not item in ['node', 'SEP', 'Reducer']:
#            routine_list.append(item)
#    return(dict_value, routine_list )
#
#def get_dict_and_routine_df(ofp):
#    #import pandas 
#    lis = list()
#    for line in open(ofp, 'r'): # 改行文字
#        lis.append(line.split()) # 区切り文字
#    else:
#        df_01 = pandas.DataFrame(lis).dropna(axis=0) # 正しい表にならない部分は DROP
#        bm_h = df_01.iloc[:,0].map(lambda x:x.startswith('#')) # Header 検出
#        bm_i = df_01.iloc[0,:].map(lambda x:x.startswith('#node')) # Index 検出
#    #if [bm_h.sum(), bm_i.sum()] == [1,1]: # '#node'を残さない
#    #    df_02 = df_01[np.logical_not(bm_h)].transpose()[np.logical_not(bm_i)].transpose()
#    #    df_02.columns = df_01[bm_h].transpose()[np.logical_not(bm_i)].iloc[:,0]
#    #    df_02.index = df_01[np.logical_not(bm_h)].transpose()[bm_i].transpose().iloc[:,0]
#    if [bm_h.sum(), bm_i.sum()] == [1,1]: # '#node'を残す
#        df_02 = df_01[np.logical_not(bm_h)]
#        df_02.columns = df_01[bm_h].transpose().iloc[:,0]
#        df_02.index = df_01[np.logical_not(bm_h)].transpose()[bm_i].transpose().iloc[:,0]
#    else:
#        df_02 = df_01.reset_index(drop=True)
#    return(df_02)
#    # usage ; df = get_dict_and_routine_df(ofp)
#
#    
#def save_txt_f1(sfp, x_true, trace_list, j_list, T_list): # sfp ; Save File Path
#    f1 = open(sfp, "w") # f1 = open('./%s/trace.txt' % i,"w")
#    f1.write("#j c1 c2 c3")
#    for P in x_true:
#        f1.write(" %s"% int(P))
#    f1.write("\n")
#    for C in range(len(trace_list[0])):
#        f1.write(str(j_list[C]))
#        f1.write(" ")
#        f1.write(str(trace_list[0][C]))
#        f1.write(" ")
#        f1.write(str(trace_list[1][C]))
#        f1.write(" ")
#        f1.write(str(trace_list[2][C]))
#        for CC in range(len(T_list)):
#            f1.write(" ")
#            f1.write(str(T_list[CC][C]))
#        f1.write("\n")
#    f1.close()
#    return
#
#def save_txt_f2(sfp, x_true, y_true, y_pre, y_pre_min, y_pre_max): # sfp ; Save File Path
#    f2 = open(sfp,'w') # f2 = open('./%s/text.txt' % i,'w')
#    f2.write("#x_true y_true y_pre y_pre_min y_pre_max")
#    f2.write("\n")
#    for k in range(len(x_true)):
#        f2.write(str(x_true[k]))
#        f2.write(" ")
#        f2.write(str(y_true[k]))
#        f2.write(" ")
#        f2.write(str(y_pre[k]))
#        f2.write(" ")
#        f2.write(str(y_pre_min[k]))
#        f2.write(" ")
#        f2.write(str(y_pre_max[k]))
#        f2.write("\n")
#    f2.close()
#    return
#
#def save_txt_f3(sfp, x_true, combined): # sfp ; Save File Path
#    f3 = open(sfp, "w") #     f3 = open('./text_predict.txt',"w")
#    f3.write("#x_true combined(added ALL)")
#    f3.write("\n")
#    for kk in range(len(x_true)):
#        f3.write(str(x_true[kk]))
#        f3.write(" ")
#        f3.write(str(combined[kk]))
#        f3.write("\n")
#    f3.close()
#    return
#
#def values_at(lis, listed_index):
#    return([lis[n] for n in listed_index]) 
#    # http://lightson.dip.jp/zope/ZWiki/099_e9_85_8d_e5_88_97_e3_81_8b_e3_82_89_e8_a4_87_e6_95_b0_e3_81_ae_e8_a6_81_e7_b4_a0_e3_82_92_e4_b8_80_e5_ba_a6_e3_81_ab_e5_8f_96_e5_be_97
#
#
#
#
#def plotting(plt, path, title, y_label, node, y_range):
#    plt.xlim(node[0], node[-1])
#    plt.xticks(node)
#
#    if not y_range[1] == 'default':
#        plt.ylim(y_range[0], y_range[1])
#
#    plt.legend()
#    plt.xscale('log', basex=2)
#    plt.yscale('log', basey=10)
#
#    plt.xlabel("Number of CPU (P)")
#    plt.ylabel('Elapse Time %s [sec]' % y_label)
#
#    plt.title('%s' % title)
#    plt.grid(which='major', linestyle='-')
#    plt.grid(which='minor', linestyle=':')
#
#    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#    plt.subplots_adjust(right=0.5)
#    plt.savefig('%s' % path, dpi=300, bbox_inches='tight')
#    plt.clf()
#    plt.close()







##from models_for_BayesianPerformancePredictionTools import test_model_151
#
#
#
#
#
#
#
#
## load parameters END #----------
#
#
#dict_value, routine_list = get_dict_and_routine(ofp)
#
#
#
## http://nbviewer.jupyter.org/github/tttamaki/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/MorePyMC.ipynb
##import pymc as pm
#
#parameter = pymc.Exponential("poisson_param", 1)
#data_generator = pymc.Poisson("data_generator", parameter)
#data_plus_one = data_generator + 1
#
#
#
#
#
##sampling = [ixnom for ixnom in nodes_in_sampling if ixnom in df.index] # 実際にdf.indexにあるものだけに限定
#sampling = [dict_value['node'].index(ixnom) for ixnom in nodes_in_sampling if ixnom in dict_value['node'] ] # 教師データとして使うエントリーのインデックスを取ってくる
##header_in_use = [colnom for colnom in header_names if colnom in df.columns]
#header_in_use = [colnom for colnom in header_names if colnom in dict_value.keys()] # 実際にdf.indexにあるものだけに限定
#
##x_true = copy.deepcopy(dict_value['node'][split_start : ])
#x_sample = copy.deepcopy(values_at(dict_value['node'], sampling)) # df.loc[sampling, '#node'].copy()
#x_sample_l = list(map(lambda x:np.log(x), x_sample ))
#x_predict = copy.deepcopy(dict_value['node']) 
#
##for i in header_in_use : # FOR文
#i = header_in_use[0]
#y_sample = copy.deepcopy(values_at(dict_value[i], sampling) ) # df.loc[sampling, i].copy()
#
#
#
#x= x_sample_l
#y= y_sample 
#
#c1 = pymc.Uniform('c1', lower=0, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
#c2 = pymc.Uniform('c2', lower=0, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
#c3 = pymc.Uniform('c3', lower=0, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
#c4 = pymc.Uniform('c4', lower=0, upper=10000000)#c4の初期分布（lowerからupperまでの一様分布）
#c5 = pymc.Uniform('c5', lower=0, upper=100000)#c5の初期分布（lowerからupperまでの一様分布）
#eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
#
#@pymc.deterministic
#def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
#    return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))
#
#@pymc.deterministic
#def function_l(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l:
#    x_list = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + (c5 * elem)/(np.exp(0.5*elem))), x ))
#    return x_list
#
#@pymc.deterministic
#def function_l100(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l: if x[i]>100:
#    term5 = 0.0 # term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
#    x_list = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + term5), x ))
#    return x_list
#
#@pymc.deterministic
#def tau(eps=eps):
#    return np.power(eps, -2)
#
#y = pymc.Normal('y', mu=function_l, tau=tau, value=y, observed=True)
#
#
#
##model = pm.Model([yes_responses, p_skewed, p])
#model = pymc.Model([y_sample, function, x_sample ])
#model = pymc.Model([y, function_l, x])
#model = pymc.Model([x])
#
#input=[mu, x]
#mcmc = pymc.MCMC(model)
#mcmc.sample(iter=100000, burn=50000, thin=10) # mcmc.sample(25000, 2500)
#

#df = get_dict_and_routine_df(ofp).astype(np.float32)
#df2 = df.set_index(['#node', 'msize'], drop=True)
#


#RFP = 'C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_3parameters_12.py' # Run File Path
#RFP = 'C:\scripts\BayesianPerformancePredictionTools-master\bayesian_3parameters_12.py' # Run File Path
#import sys, os 
#for i in list(range(10)):
#    if os.path.isdir(RFP) : sys.path.append(RFP); RFP = os.path.basename(RFP)
#    elif os.path.isfile(RFP) : RFP = os.path.basename(RFP)
#    else : break
#else:
#    runfile(RFP, wdir=os.path.basename(RFP) )
    
#    lis = list()
#    for elem in dict_value['#node']:
#        lis.append(elem  in nodes_in_sampling)
#    bm_list = list(map(lambda elem : (elem in nodes_in_sampling), dict_value['#node']))


#-----------------------------------------------------
    
    
    