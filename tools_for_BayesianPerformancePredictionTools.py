# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:44:47 2018

@author: yharada

BayesianPerformancePredictionTools で使用する自作関数とか

sys.path.append('C:\\scripts\\BayesianPerformancePredictionTools-master')
sys.path.append('C:\\scripts')

from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine
from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_df
from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_03
from tools_for_BayesianPerformancePredictionTools import save_txt_f1
from tools_for_BayesianPerformancePredictionTools import save_txt_f2
from tools_for_BayesianPerformancePredictionTools import save_txt_f3
from tools_for_BayesianPerformancePredictionTools import values_at
from tools_for_BayesianPerformancePredictionTools import plotting_02
from tools_for_BayesianPerformancePredictionTools import plotting

#from tools_for_BayesianPerformancePredictionTools import 
#from tools_for_BayesianPerformancePredictionTools import 


"""


def get_dict_and_routine(ofp):
    import collections
    f = open(ofp, 'r')
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
    return(dict_value, routine_list )
    # usage ; dict_value, routine_list = get_dict_and_routine(ofp)
    

def get_dict_and_routine_df(ofp):
    import pandas
    import numpy as np
    lis = list()
    for line in open(ofp, 'r'): # 改行文字
        lis.append(line.split()) # 区切り文字
    else:
        df_01 = pandas.DataFrame(lis).dropna(axis=0) # 正しい表にならない部分は DROP
        bm_h = df_01.iloc[:,0].map(lambda x:x.startswith('#')) # Header 検出
        bm_i = df_01.iloc[0,:].map(lambda x:x.startswith('#node')) # Index 検出
    #if [bm_h.sum(), bm_i.sum()] == [1,1]: # '#node'を残さない
    #    df_02 = df_01[np.logical_not(bm_h)].transpose()[np.logical_not(bm_i)].transpose()
    #    df_02.columns = df_01[bm_h].transpose()[np.logical_not(bm_i)].iloc[:,0]
    #    df_02.index = df_01[np.logical_not(bm_h)].transpose()[bm_i].transpose().iloc[:,0]
    if [bm_h.sum(), bm_i.sum()] == [1,1]: # '#node'を残す
        df_02 = df_01[np.logical_not(bm_h)]
        df_02.columns = df_01[bm_h].transpose().iloc[:,0]
        df_02.index = df_01[np.logical_not(bm_h)].transpose()[bm_i].transpose().iloc[:,0]
    else:
        df_02 = df_01.reset_index(drop=True)
    return(df_02)
    # usage ; df = get_dict_and_routine_df(ofp)

def get_dict_and_routine_03(ofp, set_index_names):
    import pandas
    import numpy as np
    import collections
    
    lis = list()
    for line in open(ofp, 'r'): # 改行文字
        lis.append(line.split()) # 区切り文字
    else:
        df_01 = pandas.DataFrame(lis).dropna(axis=0) # 正しい表にならない部分は DROP
        bm_h = df_01.iloc[:,0].map(lambda x:x.startswith('#')) # Header 検出
        bm_i = df_01.iloc[0,:].map(lambda x:x.startswith('#node')) # Index 検出
    
    if [bm_h.sum(), bm_i.sum()] == [1,1]: # '#node'を残す
        df_02 = df_01[np.logical_not(bm_h)].astype(np.float32)
        df_02.columns = df_01[bm_h].transpose().iloc[:,0]
        #df_02.index = df_01[np.logical_not(bm_h)].transpose()[bm_i].transpose().iloc[:,0]
    else:
        try:
            df_02 = df_01.reset_index(drop=True).astype(np.float32)
        except:
            print('skip pandas Dataframe as_type(numpy.float32)')
            df_02 = df_01.reset_index(drop=True)
        df_02 = df_01.reset_index(drop=True).astype(np.float32)
    
    try:
        # df_02 = df_02.set_index(set_index_names, drop=True)
        df_02 = df_02.set_index(set_index_names, drop=False)
    except:
        print('skip pandas Dataframe set_index')
    
    routine_list = list() # 整形する
    dict_value = collections.OrderedDict()  # 整形する
    col_names = [colnom for colnom in df_02.columns if colnom not in ['node', 'SEP', 'Reducer']] # 当該Colnomがdf.columnsにあったら除外
    #for ixnom in df_02.index:
    for colnom in col_names :
        routine_list.append(colnom )
        tmp_list = list(map(lambda ixnom:df_02.loc[ixnom,colnom], df_02.index))
        dict_value.update({colnom : tmp_list})
    return(dict_value, routine_list) # return(df_02)
    # usage ; df = get_dict_and_routine_03(ofp, ['#node', 'msize'])
    



def save_txt_f1(sfp, x_true, trace_list, j_list, T_list): # sfp ; Save File Path
    f1 = open(sfp, "w") # f1 = open('./%s/trace.txt' % i,"w")
    f1.write("#j c1 c2 c3")
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
        for CC in range(len(T_list)):
            f1.write(" ")
            f1.write(str(T_list[CC][C]))
        f1.write("\n")
    f1.close()
    return


def save_txt_f2(sfp, x_true, y_true, y_pre, y_pre_min, y_pre_max): # sfp ; Save File Path
    f2 = open(sfp,'w') # f2 = open('./%s/text.txt' % i,'w')
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
    return


def save_txt_f3(sfp, x_true, combined): # sfp ; Save File Path
    f3 = open(sfp, "w") #     f3 = open('./text_predict.txt',"w")
    f3.write("#x_true combined(added ALL)")
    f3.write("\n")
    for kk in range(len(x_true)):
        f3.write(str(x_true[kk]))
        f3.write(" ")
        f3.write(str(combined[kk]))
        f3.write("\n")
    f3.close()
    return

def values_at(lis, listed_index):
    return([lis[n] for n in listed_index]) 
    # http://lightson.dip.jp/zope/ZWiki/099_e9_85_8d_e5_88_97_e3_81_8b_e3_82_89_e8_a4_87_e6_95_b0_e3_81_ae_e8_a6_81_e7_b4_a0_e3_82_92_e4_b8_80_e5_ba_a6_e3_81_ab_e5_8f_96_e5_be_97


def plotting_02(path, title, y_label, node, y_range):
    import matplotlib.pyplot as plt
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
    return
    
    
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













#-----------------------------------------------------
#import pymc
#import numpy as np
#import pandas
#import matplotlib.pyplot as plt
# import copy, math, os, sys, collections, random, argparse
#-----------------------------------------------------


