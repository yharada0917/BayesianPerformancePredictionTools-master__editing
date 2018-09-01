# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:40:51 2016

Work OK 180821
# Modelを関数化して実行してみる 5 Params
@author: harada


#runfile('C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_5parameters_01b.py', 
#        wdir='C:\\scripts\\BayesianPerformancePredictionTools-master')

#runfile('C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_5parameters_13.py', 
#        wdir='C:\\scripts\\BayesianPerformancePredictionTools-master')

runfile('C:\\scripts\\BayesianPerformancePredictionTools-master\\bayesian_5PM_parameters_04_fakedata.py', 
        wdir='C:\\scripts\\BayesianPerformancePredictionTools-master')
"""


import pymc
import numpy as np
import pandas
import matplotlib.pyplot as plt
import copy, math, os, sys, collections, random, argparse
import time






def main_13():
    
    # モジュールのインポート #-----------------------------------------------------
    started_at = time.time() # 時間を測りたい処理 start
    ## いちおうパスを追加しておく
    #sys.path.append('C:\\scripts')
    #sys.path.append('C:\\scripts\\BayesianPerformancePredictionTools-master')
    
    from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_03
    from tools_for_BayesianPerformancePredictionTools import mesh_3D_data_from_stacked3obj
    #from tools_for_BayesianPerformancePredictionTools import values_at
    
    from models_for_BayesianPerformancePredictionTools import test_model_255
    from models_for_BayesianPerformancePredictionTools import LogT_cost_255
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # モジュールのインポート END #-----------------------------------------------------
    
    
    
    # 事前準備 #-----------------------------------------------------
    #ofp = 'C:\scripts\BayesianPerformancePredictionTools-master\elapse_time_table.txt' # open file path
    #dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node']) # dict_value, routine_list = get_dict_and_routine(ofp)
    
    ofp = 'elapse_time_table_pm_fake.txt' # open file path # elapse_time_table_pm
    #current_pickle_nom= 'for_current_pymc.pickle'
    dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node', 'msize']) # dict_value, routine_list = get_dict_and_routine(ofp)
    
    
    nodes_in_sampling = [10,100] # MCMC教師として使用するNode数 # for dict version
    msize_in_sampling = [10000,100000] # MCMC教師として使用するNode数 # for dict version
    parameter = ['c1','c2','eps','tau']  # P+M 5params
    header_names = ['fake01', 'fake02', 'fake03', 'fake04', 'fake05', 'fake06', 'fake07'] # MCMCとして使用するHeader
    
    
    bm_list_1 = list(map(lambda elem : elem in nodes_in_sampling, dict_value['#node'])) # 条件の一致するものをTrue
    bm_list_2 = list(map(lambda elem : elem in msize_in_sampling, dict_value['msize'])) # 条件の一致するものをTrue
    bm_list = np.logical_and(bm_list_1,bm_list_2) # sampling = np.where(bm_list==True)
    header_in_use = [colnom for colnom in header_names if colnom in dict_value.keys()] # 実際にdf.indexにあるものだけに限定
    
    
    xp_sample = np.array(dict_value['#node'])[bm_list]
    xp_sample_l = np.log(xp_sample).copy()
    xm_sample = np.array(dict_value['msize'])[bm_list]
    xm_sample_l = np.log(xm_sample).copy()
    
    xp_val = np.array(dict_value['#node'])
    xp_val_l = np.log(xp_val).copy()
    xm_val = np.array(dict_value['msize'])
    xm_val_l = np.log(xm_val).copy()
    df_dict_value = pandas.DataFrame(dict_value, index=[z for z in zip(xp_val, xm_val)])


    print('[prepare ] '+str(time.time() - started_at)+' (s) ')
    # 事前準備 END #-----------------------------------------------------
    
    
    
    
    for i in header_in_use : # FOR文 
        # i = header_in_use[0] # Testrun
        
        
        # 本体 #-----------------------------------------------------
        y_sample = np.array(dict_value[i])[bm_list]
        y_sample_l = np.log(y_sample).copy()
        
        print(i, y_sample_l ) # ; print()
        current_pickle_nom = os.path.join(str(i), 'for_current_pymc.pickle') # 保存先文字列合成 # 'for_current_pymc.'+str(i)+'.pickle'
        os.system('mkdir %s' % i)
        
        
        sampler = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l), db='pickle', dbname=current_pickle_nom) # https://pymc-devs.github.io/pymc/database.html
        sampler.sample(iter=1000, burn=0, thin=2)
        
        
        trace_d = dict() # trace_list = list()
        for j in parameter:
            mcmctrace = sampler.db.trace(j)[:] # sampler.db.trace('c1')[:])はnumpy.ndarrayを返す
            trace_d.update({ j:mcmctrace }) # trace_list.append(mcmctrace)
            pymc.Matplot.plot(sampler.db.trace(j))
            pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
        sampler.db.close()
        
        print('[iter for '+ i +' ] '+  str(time.time() - started_at)+' (s) ')
        # 本体 END #-----------------------------------------------------
        
        
        
        # 計算された所要時間・結果の整理 #-----------------------------------------------------
        y_val = np.array(dict_value[i])
        y_val_l = np.log(y_val).copy()
        
        res_list = list()
        res_list_b = list()
        for c1,c2,eps,tau in zip(trace_d['c1'],trace_d['c2'],trace_d['eps'],trace_d['tau']):
            y_predict_l = LogT_cost_255(xp_val_l, xm_val_l, c1, c2) # c1,c2はFloatでなくてはならない # 返り値はLogT
            res_list.append(np.hstack([y_predict_l, c1, c2, eps, tau]))
            res_list_b.append(np.hstack([np.exp(y_predict_l), c1, c2, eps, tau]))
        else:
            df_trace_predict_l = pandas.DataFrame(res_list, columns=[z for z in zip(xp_val_l, xm_val_l)]+parameter ) # DF column names はListで渡す
            df_trace_predict = pandas.DataFrame(res_list_b, columns=[z for z in zip(xp_val, xm_val)]+parameter ) # DF column names はListで渡す
        # 計算された所要時間・結果の整理 END #-----------------------------------------------------
        
        
        
        # 3D Plot #-----------------------------------------------------
        plot_range = [[1,6],[11,16],[41,46],[251,256]]
        
        # 3D Plot 専用　計算された所要時間・結果の整理 #
        res_list_c = list()
        res_list_d = list()
        for c1,c2,eps,tau in zip(trace_d['c1'],trace_d['c2'],trace_d['eps'],trace_d['tau']):
            y_predict_l = LogT_cost_255(xp_val_l, xm_val_l, c1, c2) # c1,c2はFloatでなくてはならない # 返り値はLogT
            res_list_c.append(y_predict_l)
            res_list_d.append(np.exp(y_predict_l)) # np.hstack([np.exp(y_predict_l), c1, c2, eps, tau]))
        #else:
        #    df_trace_predict_l = pandas.DataFrame(res_list_c, columns=[z for z in zip(xp_val_l, xm_val_l)]) # DF column names はListで渡す
        #    df_trace_predict = pandas.DataFrame(res_list_d, columns=[z for z in zip(xp_val, xm_val)]) # DF column names はListで渡す
        # 3D Plot 専用　計算された所要時間・結果の整理 END #
        
        
        for [ii,jj] in plot_range :
            plotname = '3D_map_'+str(ii)+'_to_'+str(jj)+'.png'
            red_plot_list = list(range(ii,jj,1)) # [6,11,16,21]
            #for ix in red_plot_list :
            #    y_predict_l = res_list_c[ix]
            
            
            [X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_val_l)
            [X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
            
            fig = plt.figure()
            ax = Axes3D(fig)
            #ax.plot_wireframe([xp_val_l, xm_val_l, y_predict_l]) #<---ここでplot
            #ax.plot_wireframe(df_unstack.index, df_unstack.columns, np.array(df_unstack)) #<---ここでplot
            #ax.plot_wireframe(mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l) ) #<---ここでplot
            ax.plot_wireframe(X,Y,Z, color='b') #<---ここでplot
            # ax.plot_wireframe(xp_val_l, xm_val_l, y_val_l, color='b') #<---ここでplot
            for ix in red_plot_list :
                y_predict_l = res_list_c[ix]
                ax.scatter3D(xp_val_l, xm_val_l, y_predict_l, color='r')
            #ax.set_xlim([max(X), min(X)]) # 負→正の軸を逆転させたいとき
            #ax.set_ylim([max(Y), min(Y)]) # 負→正の軸を逆転させたいとき
            #ax.xscale('log') ; ax.yscale('log') ; ax.zscale('log') # NOT work # https://teratail.com/questions/124817
            ax.set_xlabel("LOG(P)")
            ax.set_ylabel("LOG(M)")
            ax.set_zlabel("LOG(T)")
            #plt.show()
            plt.savefig(os.path.join(i,plotname))
        # 3D Plot END #-----------------------------------------------------            
        
        # 結果の保存 #-----------------------------------------------------
        sfp = os.path.join(str(i), 'trace_and_predict_0.txt') # 保存先文字列合成
        df_trace_predict.to_csv(sfp, header=True, index=True)
        sfp = os.path.join(str(i), 'trace_and_predict_1_log.txt') # 保存先文字列合成
        df_trace_predict_l.to_csv(sfp, header=True, index=True)
        sfp = os.path.join(str(i), 'original.txt') # 保存先文字列合成
        df_dict_value.to_csv(sfp, header=True, index=True)
        # 結果の保存 END #-----------------------------------------------------
        
    del(started_at)
    return 0



#-----------------------------------------------------
if __name__ == '__main__':
    main_13()
else:
    print('skip run "__main__"')
#-----------------------------------------------------













# # # ここまでが確認ずみ作業

# # # 以下は下書き
#
#plot_range = [[1,6],[11,16],[41,46],[81,86]]
#
#red_plot_list = list(range(1,6,1)) # [6,11,16,21]
##for ix in red_plot_list :
##    y_predict_l = res_list_c[ix]
#
#
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_val_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#
#fig = plt.figure()
#ax = Axes3D(fig)
##ax.plot_wireframe([xp_val_l, xm_val_l, y_predict_l]) #<---ここでplot
##ax.plot_wireframe(df_unstack.index, df_unstack.columns, np.array(df_unstack)) #<---ここでplot
##ax.plot_wireframe(mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l) ) #<---ここでplot
#ax.plot_wireframe(X,Y,Z, color='b') #<---ここでplot
## ax.plot_wireframe(xp_val_l, xm_val_l, y_val_l, color='b') #<---ここでplot
#for ix in red_plot_list :
#    y_predict_l = res_list_c[ix]
#    ax.scatter3D(xp_val_l, xm_val_l, y_predict_l, color='r')
##ax.set_xlim([max(X), min(X)]) # 負→正の軸を逆転させたいとき
##ax.set_ylim([max(Y), min(Y)]) # 負→正の軸を逆転させたいとき
##ax.xscale('log') ; ax.yscale('log') ; ax.zscale('log') # NOT work # https://teratail.com/questions/124817
#ax.set_xlabel("LOG(P)")
#ax.set_ylabel("LOG(M)")
#ax.set_zlabel("LOG(T)")
#plt.show()
#
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#
#from tools_for_BayesianPerformancePredictionTools import mesh_3D_data_from_stacked3obj
#from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_03
#from models_for_BayesianPerformancePredictionTools import LogT_cost_255
#
#
#dirname = 'fake01'
#i = 'fake01'
#ofp = 'elapse_time_table_pm_fake.txt' # open file path # elapse_time_table_pm
#dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node', 'msize']) # dict_value, routine_list = get_dict_and_routine(ofp)
#parameter = ['c1','c2','eps','tau']  # P+M 5params
#
##res_df = pandas.read_csv('fake01\\trace_and_predict_0.txt')
##res_df_l = pandas.read_csv('fake01\\trace_and_predict_1_log.txt')
##colnom_list = list(res_df.columns)
#
#y_val = np.array(dict_value[i])
#y_val_l = np.log(y_val).copy()
#
#
#xp_val = np.array(dict_value['#node'])
#xp_val_l = np.log(xp_val).copy()
#xm_val = np.array(dict_value['msize'])
#xm_val_l = np.log(xm_val).copy()
#
#
#db = pymc.database.pickle.load(os.path.join(dirname, 'for_current_pymc.pickle'))
##db = pymc.database.pickle.load(os.path.join(dirname, 'for_current_pymc.pickle'))
##db = pymc.database.pickle.load('fake01\\for_current_pymc.pickle')
#
#trace_d = dict() # trace_list = list()
#for j in parameter:
#    mcmctrace = db.trace(j)[:] # sampler.db.trace('c1')[:])はnumpy.ndarrayを返す
#    trace_d.update({ j:mcmctrace }) # trace_list.append(mcmctrace)
#    #pymc.Matplot.plot(db.trace(j))
#    #pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
#db.close()


#
#res_list_c = list()
#res_list_d = list()
#for c1,c2,eps,tau in zip(trace_d['c1'],trace_d['c2'],trace_d['eps'],trace_d['tau']):
#    y_predict_l = LogT_cost_255(xp_val_l, xm_val_l, c1, c2) # c1,c2はFloatでなくてはならない # 返り値はLogT
#    res_list_c.append(y_predict_l)
#    res_list_d.append(np.exp(y_predict_l)) # np.hstack([np.exp(y_predict_l), c1, c2, eps, tau]))
##else:
##    df_trace_predict_l = pandas.DataFrame(res_list_c, columns=[z for z in zip(xp_val_l, xm_val_l)]) # DF column names はListで渡す
##    df_trace_predict = pandas.DataFrame(res_list_d, columns=[z for z in zip(xp_val, xm_val)]) # DF column names はListで渡す
#
#
#red_plot_list = list(range(501,510,1)) # [6,11,16,21]
##for ix in red_plot_list :
##    y_predict_l = res_list_c[ix]
#
#
#
#
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_val_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#
#
#fig = plt.figure()
#ax = Axes3D(fig)
##ax.plot_wireframe([xp_val_l, xm_val_l, y_predict_l]) #<---ここでplot
##ax.plot_wireframe(df_unstack.index, df_unstack.columns, np.array(df_unstack)) #<---ここでplot
##ax.plot_wireframe(mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l) ) #<---ここでplot
#ax.plot_wireframe(X,Y,Z, color='b') #<---ここでplot
## ax.plot_wireframe(xp_val_l, xm_val_l, y_val_l, color='b') #<---ここでplot
#for ix in red_plot_list :
#    y_predict_l = res_list_c[ix]
#    ax.scatter3D(xp_val_l, xm_val_l, y_predict_l, color='r')
##ax.set_xlim([max(X), min(X)]) # 負→正の軸を逆転させたいとき
##ax.set_ylim([max(Y), min(Y)]) # 負→正の軸を逆転させたいとき
##ax.xscale('log') ; ax.yscale('log') ; ax.zscale('log') # NOT work # https://teratail.com/questions/124817
#ax.set_xlabel("LOG(P)")
#ax.set_ylabel("LOG(M)")
#ax.set_zlabel("LOG(T)")
#plt.show()












##res_list = list()
##res_list_b = list()
#listed_ps = list()
#ps = pandas.Series(y_val_l, index=zip(xp_val_l, xm_val_l))
#listed_ps.append(ps)
#for ix in red_plot_list :
#    listed_ps.append(df_trace_predict_l.iloc[ix,:])
#else:
#    df = pandas.concat(listed_ps, axis=1, sort=True).dropna(axis=0)
#
#for colnom in red_plot_list : # df.columns:
#    ps = df[colnom]
#    
#
#
#
#
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val, xm_val, y_val)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#    
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val, xm_val, y_val_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#
#
#
#
#
#
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot_wireframe(X,Y,Z, c='b') #<---ここでplot
#plt.show()
#
#
#
#
#
#
#len(db.trace('c1')[:])
#
#
#df_unstack = pandas.DataFrame([xp_val_l, xm_val_l, y_predict_l]).T.set_index([0,1]).unstack()
#
#df_unstack = pandas.DataFrame([xp_val_l, xm_val_l, y_predict_l]).T.set_index([0]).set_columns([1])
#
#fig = plt.figure()
#ax = Axes3D(fig)
##ax.plot_wireframe([xp_val_l, xm_val_l, y_predict_l]) #<---ここでplot
##ax.plot_wireframe(df_unstack.index, df_unstack.columns, np.array(df_unstack)) #<---ここでplot
##ax.plot_wireframe(mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l) ) #<---ここでplot
#ax.plot_wireframe(X,Y,Z) #<---ここでplot
#
#plt.show()
#            
#
#
#
#            
#            
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val, xm_val, y_val)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#    
#[X,Y,Z] = mesh_3D_data_from_stacked3obj(xp_val, xm_val, y_val_l)
#[X,Y,Z] = [np.array(X),np.array(Y),np.array(Z)]
#    
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot_wireframe(X,Y,Z, c='b') #<---ここでplot
#plt.show()

#
#    df = pandas.DataFrame(columns=colmon_list, index=ixmon_list )
#    for x,y,z in zip(xp_val_l, xm_val_l, y_predict_l):
#        df.loc[x,y] = z
#        
#    #for colnom in colmon_list:
#    #    for colnom in colmon_list:
#    #        df.loc[ixnom,colnom] = 
#        













#
## 簡易版のサマリーの出力がPyMCにある模様。Trace は保存不可の模様
#import pandas
#sfp = "save_mcmc_summary.csv"
#sampler.write_csv(sfp, variables=parameter) # variables=['c1','c2','eps','tau'] # https://pymc-devs.github.io/pymc/database.html
#hogehoge = pandas.read_csv(sfp)
#
## sampler that instead will write data to a pickle file:
#current_pickle_nom= 'for_current_pymc.pickle'
#sampler = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l), db='pickle', dbname=current_pickle_nom) # sampler = MCMC(disaster_model, db='pickle', dbname=current_pickle_nom)
## sampler.use_step_method(pymc.AdaptiveMetropolis, [c1,c2,eps,tau])
#sampler.sample(iter=10000, burn=10, thin=10)
#
#sampler.db.close()
#
#hogehoge = sampler.db.trace('c1')[:]
#



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



#def main_13():
#    
#    # モジュールのインポート #-----------------------------------------------------
#    started_at = time.time() # 時間を測りたい処理 start
#    ## いちおうパスを追加しておく
#    #sys.path.append('C:\\scripts')
#    #sys.path.append('C:\\scripts\\BayesianPerformancePredictionTools-master')
#    
#    #from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine
#    #from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_df
#    from tools_for_BayesianPerformancePredictionTools import get_dict_and_routine_03
#    #from tools_for_BayesianPerformancePredictionTools import save_txt_f1
#    #from tools_for_BayesianPerformancePredictionTools import save_txt_f2
#    #from tools_for_BayesianPerformancePredictionTools import save_txt_f3
#    from tools_for_BayesianPerformancePredictionTools import values_at
#    #from tools_for_BayesianPerformancePredictionTools import plotting_02
#    #from tools_for_BayesianPerformancePredictionTools import plotting
#    
#    from models_for_BayesianPerformancePredictionTools import test_model_153
#    from models_for_BayesianPerformancePredictionTools import test_model_251
#    from models_for_BayesianPerformancePredictionTools import test_model_252
#    from models_for_BayesianPerformancePredictionTools import test_model_253
#    from models_for_BayesianPerformancePredictionTools import test_model_253l
#    from models_for_BayesianPerformancePredictionTools import test_model_253lb
#    from models_for_BayesianPerformancePredictionTools import test_model_254
#    from models_for_BayesianPerformancePredictionTools import test_model_255
#    # モジュールのインポート END #-----------------------------------------------------
#    
#    
#    
#    # 事前準備 #-----------------------------------------------------
#    #ofp = 'C:\scripts\BayesianPerformancePredictionTools-master\elapse_time_table.txt' # open file path
#    #dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node']) # dict_value, routine_list = get_dict_and_routine(ofp)
#    
#    
#    ofp = 'elapse_time_table_pm_fake.txt' # open file path # elapse_time_table_pm
#    dict_value, routine_list = get_dict_and_routine_03(ofp, ['#node', 'msize']) # dict_value, routine_list = get_dict_and_routine(ofp)
#    
#    
#    nodes_in_sampling = [10,100,1000] # MCMC教師として使用するNode数 # for dict version
#    msize_in_sampling = [10000,100000,1000000] # MCMC教師として使用するNode数 # for dict version
#    #parameter = ['c1','c2','c3','eps','tau'] # 3Params
#    #parameter = ['c1','c2','c3','c4','c5','eps','tau'] # 5params
#    #parameter = ['c11','c21','c31','c41','c51', 
#    #             'c12','c22','c32','c42','c52','eps','tau']  # P+M 5params
#    #parameter = ['c10','c20','c30','c40','c50',
#    #             'c11','c21','c31','c41','c51',
#    #             'c12','c22','c32','c42','c52','eps','tau']  # P+M 5params
#    parameter = ['c1','c2','eps','tau']  # P+M 5params
#    header_names = ['fake01', 'fake02', 'fake03'] # MCMCとして使用するHeader
#    #header_names = ['Total', 'pdormtr', 'pdstedc', 'pdsytrd', 'pdsygst', 'pdpotrf', 'rest'] # MCMCとして使用するHeader
#    
#    
#    bm_list_1 = list(map(lambda elem : elem in nodes_in_sampling, dict_value['#node'])) # 条件の一致するものをTrue
#    bm_list_2 = list(map(lambda elem : elem in msize_in_sampling, dict_value['msize'])) # 条件の一致するものをTrue
#    bm_list = np.logical_and(bm_list_1,bm_list_2) # sampling = np.where(bm_list==True)
#    # sampling = [dict_value['#node'].index(ixnom) for ixnom in nodes_in_sampling if ixnom in dict_value['#node'] ] # 教師データとして使うエントリーのインデックスを取ってくる
#    header_in_use = [colnom for colnom in header_names if colnom in dict_value.keys()] # 実際にdf.indexにあるものだけに限定
#    
#    
#    xp_sample = np.array(dict_value['#node'])[bm_list].copy()
#    xp_sample_l = np.log(xp_sample).copy()
#    #xp_predict = np.array(dict_value['#node']).copy()
#    #
#    xm_sample = np.array(dict_value['msize'])[bm_list].copy()
#    xm_sample_l = np.log(xm_sample).copy()
#    #xm_predict = np.array(dict_value['msize']).copy()
#    
#
#    #xp_sample = copy.deepcopy(list(np.array(dict_value['#node'])[bm_list]))
#    #xm_sample = copy.deepcopy(list(np.array(dict_value['msize'])[bm_list]))
#
#
#
#
#
#    #x_sample = copy.deepcopy(values_at(dict_value['#node'], sampling)) # df.loc[sampling, '#node'].copy()
#    #x_sample_l = list(map(lambda x:np.log(x), x_sample ))
#    #x_predict = copy.deepcopy(dict_value['#node']) 
#    print('[prepare ] '+str(time.time() - started_at)+' (s) ')
#    # 事前準備 END #-----------------------------------------------------
#    
#    
#    
#    #x_sample = np.array([xp_sample,xm_sample]).T
#    
#    # 本体 #-----------------------------------------------------
#    for i in header_in_use : # FOR文
#        # i = header_in_use[0]
#        # y_sample = np.array(dict_value[i])[bm_list].copy()
#        y_sample = copy.deepcopy(list(np.array(dict_value[i])[bm_list]))
#        y_sample_l = np.log(y_sample).copy()
#        print(i, y_sample_l ) # ; print()
#        #print(i, values_at(dict_value[i], sampling)) # ; print()
#        #y_sample = copy.deepcopy(values_at(dict_value[i], sampling))
#        
#        #mcmc = pymc.MCMC(test_model_153(xp_sample_l, y_sample))
#        #mcmc = pymc.MCMC(test_model_251(xp_sample, xm_sample, y_sample))
#        #mcmc = pymc.MCMC(test_model_253(xp_sample, xm_sample, y_sample)) # 簡易モデル
#        #mcmc = pymc.MCMC(test_model_253l(xp_sample, xm_sample, y_sample))
#        #mcmc = pymc.MCMC(test_model_253lb(x_sample, y_sample))
#        #mcmc = pymc.MCMC(test_model_254(xp_sample, xm_sample, y_sample))
#        mcmc = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l))
#        mcmc.sample(iter=200000, burn=100, thin=10)
#        
#        os.system('mkdir %s' % i) # 出力先DIR作成
#        trace_list = []
#        for j in parameter:
#            pymc.Matplot.plot(mcmc.trace(j))
#            mcmctrace = np.array(mcmc.trace("%s" % j, chain=None)[:])
#            # print("mcmctrace:",mcmctrace)
#            trace_list.append(mcmctrace)
#            pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
#            plt.clf()
#            plt.close()
#        
#        print('[iter for '+ i +' ] '+  str(time.time() - started_at)+' (s) ')
#    
#    ##for i in header_in_use : # FOR文
#    #i = header_in_use[0]
#    #y_sample = copy.deepcopy(values_at(dict_value[i], sampling))
#    #
#    #mcmc = pymc.MCMC(test_model_153(np.array(x_sample_l), y_sample))
#    #mcmc.sample(iter=100000, burn=50000, thin=10)
#    #
#    #trace_list = []
#    #for j in parameter:
#    #    pymc.Matplot.plot(mcmc.trace(j))
#    #    mcmctrace = np.array(mcmc.trace("%s" % j, chain=None)[:])
#    #    print("mcmctrace:",mcmctrace)
#    #    trace_list.append(mcmctrace)
#    #    pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
#    #    plt.clf()
#    #    plt.close()
#    # 本体 END #-----------------------------------------------------
#    del(started_at)
#    return 0

#
##-----------------------------------------------------
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description = '')
#    parser.add_argument('-yr', '--y_range',
#                        nargs = 2,
#                        type = float,
#                        default = [10, 'default'],
#                        help = 'y range (min max)')
#
#    parser.add_argument('-l', '--log',
#                        action = 'store_true',
#                        help = 'log flag (default = False)')
#
#    args = parser.parse_args()
#    y_range = args.y_range
#    l = args.log
#    main_13()
#else:
#    print('skip run "__main__"')
    
# # OLD
#    # 本体 #-----------------------------------------------------
#    for i in header_in_use : # FOR文 
#        # i = header_in_use[0] # Testrun
#        y_sample = np.array(dict_value[i])[bm_list]
#        y_sample_l = np.log(y_sample).copy()
#        print(i, y_sample_l ) # ; print()
#        current_pickle_nom= 'for_current_pymc.'+str(i)+'.pickle'
#
#
#        sampler = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l), db='pickle', dbname=current_pickle_nom) # https://pymc-devs.github.io/pymc/database.html
#        sampler.sample(iter=10000, burn=10, thin=10)
#        #sampler = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l))
#        #sampler.sample(iter=200000, burn=100, thin=10)
#
#        trace_list = []
#        for j in parameter:
#            mcmctrace = sampler.db.trace(j)[:] # sampler.db.trace('c1')[:])はnumpy.ndarrayを返す
#            trace_list.append(mcmctrace)
#            pymc.Matplot.plot(sampler.db.trace(j))
#        
#        #os.system('mkdir %s' % i) # 出力先DIR作成
#        #trace_list = []
#        #for j in parameter:
#        #    pymc.Matplot.plot(sampler.trace(j))
#        #    mcmctrace = np.array(sampler.trace("%s" % j, chain=None)[:])
#        #    # print("mcmctrace:",mcmctrace)
#        #    trace_list.append(mcmctrace)
#        #    pymc.Matplot.savefig('./%s/graph_%s.png' % (i, j))
#        #    plt.clf()
#            plt.close()
#        
#        sampler.db.close()
#        
#        print('[iter for '+ i +' ] '+  str(time.time() - started_at)+' (s) ')
#    # 本体 END #-----------------------------------------------------    
    
    
#        y_val = np.array(dict_value[i])
#        y_val_l = np.log(y_val).copy()
#        
#        res_list = list()
#        for c1,c2,eps,tau in zip(trace_d['c1'],trace_d['c2'],trace_d['eps'],trace_d['tau']):
#            # c1,c2,eps,tau = trace_d['c1'][0],trace_d['c2'][0],trace_d['eps'][0],trace_d['tau'][0] # TEST
#            # y_predict_l = LogT_cost_255(xp_val_l, xm_val_l, c1, c2) # c1,c2はFloatでなくてはならない
#            # print(c1,c2,eps,tau )
#            y_predict_l = LogT_cost_255(xp_val_l, xm_val_l, c1, c2) # c1,c2はFloatでなくてはならない # 返り値はLogT
#            res_list.append(np.hstack([y_predict_l, c1, c2, eps, tau]))
#        else:
#            df_trace_predict_l = pandas.DataFrame(res_list, columns=[z for z in zip(xp_val_l, xm_val_l)]+parameter ) # DF column names はListで渡す
#
#        res_list = list()
#        for c1,c2,eps,tau in zip(trace_d['c1'],trace_d['c2'],trace_d['eps'],trace_d['tau']):
#            # c1,c2,eps,tau = trace_d['c1'][0],trace_d['c2'][0],trace_d['eps'][0],trace_d['tau'][0] # TEST
#            # y_predict_l = LogT_cost_255(xp_val_l, xm_val_l, c1, c2) # c1,c2はFloatでなくてはならない
#            # print(c1,c2,eps,tau )
#            y_predict = np.exp(LogT_cost_255(xp_val_l, xm_val_l, c1, c2)) # c1,c2はFloatでなくてはならない # 返り値はLogT
#            res_list.append(np.hstack([y_predict, c1, c2, eps, tau]))
#        else:


#def mesh_3D_data_from_stacked3obj(xp_val_l, xm_val_l, y_predict_l):
#    na_fill_obj = ''
#    colmon_list = list(set(xp_val_l))
#    ixmon_list = list(set(xm_val_l))
#    d = dict()
#    for x,y,z in zip(xp_val_l, xm_val_l, y_predict_l):
#        d.update({(x,y):z})
#    listed_lis = list()
#    for colnom in colmon_list:
#        lis = list()
#        for ixnom in ixmon_list :
#            try:
#                lis.append(d[(colnom,ixnom)])
#            except:
#                lis.append(na_fill_obj )
#        else:
#            listed_lis.append(lis )
#    return(colmon_list, ixmon_list, listed_lis)
#    # 同じ座標の入力が複数ある場合は使用しないでください List will returned

##-----------------------------------------------------


#-----------------------------------------------------
    
    
    