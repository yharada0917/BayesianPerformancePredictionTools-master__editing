# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:14:45 2018

@author: yharada

# BayesianPerformancePredictionToolsにおける"MYMODEL"


from models_for_BayesianPerformancePredictionTools import test_model_131
"""
import pymc
import numpy as np



def test_model_01():
    #　https://github.com/pymc-devs/pymc
    # Import relevant modules
    import pymc
    import numpy as np
    
    # Some data
    n = 5 * np.ones(4, dtype=int)
    x = np.array([-.86, -.3, -.05, .73])
    
    # Priors on unknown parameters
    alpha = pymc.Normal('alpha', mu=0, tau=.01)
    beta = pymc.Normal('beta', mu=0, tau=.01)
    
    # Arbitrary deterministic function of parameters
    @pymc.deterministic
    def theta(a=alpha, b=beta):
        """theta = logit^{-1}(a+b)"""
        return pymc.invlogit(a + b * x)
    
    # Binomial likelihood for data
    d = pymc.Binomial('d', n=n, p=theta, value=np.array([0., 1., 3., 5.]),
                      observed=True)
    return locals()



def test_model_02(x):
    #　https://github.com/pymc-devs/pymc
    # Import relevant modules
    import pymc
    import numpy as np
    
    # Some data
    n = 5 * np.ones(4, dtype=int)
    #x = np.array([-.86, -.3, -.05, .73])
    
    # Priors on unknown parameters
    alpha = pymc.Normal('alpha', mu=0, tau=.01)
    beta = pymc.Normal('beta', mu=0, tau=.01)
    
    # Arbitrary deterministic function of parameters
    @pymc.deterministic
    def theta(a=alpha, b=beta):
        """theta = logit^{-1}(a+b)"""
        return pymc.invlogit(a + b * x)
    
    # Binomial likelihood for data
    d = pymc.Binomial('d', n=n, p=theta, value=np.array([0., 1., 3., 5.]),
                      observed=True)
    return locals()




def test_model_132(x, y):
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=-100000, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=-100000, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
    c3 = pymc.Uniform('c3', lower=-100000, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.00001)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）

    @pymc.deterministic
    def function(x=x, c1=c1, c2=c2, c3=c3):
        return (c1 / np.exp(x)) + c2 + (c3 * np.exp(x))
    
    
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()


def test_model_133(x, y):
    #import pymc
    #import numpy as np
    
    c1 = pymc.Uniform('c1', lower=-100000, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=-100000, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
    c3 = pymc.Uniform('c3', lower=-100000, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.00001)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）

    @pymc.deterministic
    def function(x=x, c1=c1, c2=c2, c3=c3):
        return (c1 / np.exp(x)) + c2 + (c3 * np.exp(x))
    
    
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()


def test_model_151(x, y, l):
    import pymc
    import numpy as np
    
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


def test_model_152(x, y, l):
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
    c3 = pymc.Uniform('c3', lower=0, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
    c4 = pymc.Uniform('c4', lower=0, upper=10000000)#c4の初期分布（lowerからupperまでの一様分布）
    c5 = pymc.Uniform('c5', lower=0, upper=100000)#c5の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）

    @pymc.deterministic
    def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
        return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))
    
    @pymc.deterministic
    def function_l(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l:
        x_list = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + (c5 * elem)/(np.exp(0.5*elem)), x )))
        return x_list
    
    @pymc.deterministic
    def function_l100(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l: if x[i]>100:
        term5 = 0.0 # term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
        x_list = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + term5)))
        return x_list
    
    #@pymc.deterministic
    #def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
    #    if l:
    #        x_list = []
    #        for i in range(len(x)):
    #            if x[i]>100:
    #                term5 = 0.0
    #            else:
    #                term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
    #            x_list.append(np.log((c4 / np.exp(2.0*x[i])) + (c1 / np.exp(x[i])) + c2 + (c3 * x[i]) + term5))
    #        return x_list
    #    else:
    #        return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))


    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)

    y = pymc.Normal('y', mu=function_l, tau=tau, value=y, observed=True)
    return locals()



# Array で渡してみる　これでも問題なし
def test_model_153(x, y):
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
    c3 = pymc.Uniform('c3', lower=0, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
    c4 = pymc.Uniform('c4', lower=0, upper=10000000)#c4の初期分布（lowerからupperまでの一様分布）
    c5 = pymc.Uniform('c5', lower=0, upper=100000)#c5の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic
    def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
        return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))
    
    #@pymc.deterministic
    #def function_l(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l:
    #    obj = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + (c5 * elem)/(np.exp(0.5*elem))), x ))
    #    return obj
    #
    #@pymc.deterministic
    #def function_l100(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l: if x[i]>100:
    #    term5 = 0.0 # term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
    #    obj = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + term5), x ))
    #    return obj
    
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()
#Usage;
##model = pymc.Model([function_l, x])
##pymc.numpy.random.seed(0)
#mcmc = pymc.MCMC(test_model_153(np.array(x_sample_l), y_sample))
#mcmc.sample(iter=100000, burn=50000, thin=10)



def test_model_251(xp, xm, y):
    import pymc
    import numpy as np
    
    c11 = pymc.Uniform('c11', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c12 = pymc.Uniform('c12', lower=0, upper=100000)#c12の初期分布（lowerからupperまでの一様分布）
    c21 = pymc.Uniform('c21', lower=0, upper=100000)#c21の初期分布（lowerからupperまでの一様分布）
    c22 = pymc.Uniform('c22', lower=0, upper=100000)#c22の初期分布（lowerからupperまでの一様分布）
    c31 = pymc.Uniform('c31', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    c32 = pymc.Uniform('c32', lower=0, upper=100000)#c32の初期分布（lowerからupperまでの一様分布）
    c41 = pymc.Uniform('c41', lower=0, upper=10000000)#c41の初期分布（lowerからupperまでの一様分布）
    c42 = pymc.Uniform('c42', lower=0, upper=10000000)#c42の初期分布（lowerからupperまでの一様分布）
    c51 = pymc.Uniform('c51', lower=0, upper=100000)#c51の初期分布（lowerからupperまでの一様分布）
    c52 = pymc.Uniform('c52', lower=0, upper=100000)#c52の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def function(xp=xp, c11=c11, c21=c21, c31=c31, c41=c41, c51=c51,
                 xm=xm, c12=c12, c22=c22, c32=c32, c42=c42, c52=c52):
        t1 = ( c11*np.power(xm,3) + c12*np.power(xm,2) ) / xp
        t2 = ( c21*np.power(xm,3) + c22*np.power(xm,2) )
        t3 = ( c31*xm + c32 ) * np.log(xp)
        t4 = ( c41*np.power(xm,3) + c42*np.power(xm,2) ) / np.power(xp,2)
        t5 = ( c51*np.power(xm,2) + c52*xm ) * np.log(xp) / np.sqrt(xp)
        return(t1+t2+t3+t4+t5)
            
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()



def test_model_252(xp, xm, y):
    import pymc
    import numpy as np
    
    c10 = pymc.Uniform('c10', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c11 = pymc.Uniform('c11', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c12 = pymc.Uniform('c12', lower=0, upper=100000)#c12の初期分布（lowerからupperまでの一様分布）
    c20 = pymc.Uniform('c20', lower=0, upper=100000)#c21の初期分布（lowerからupperまでの一様分布）
    c21 = pymc.Uniform('c21', lower=0, upper=100000)#c21の初期分布（lowerからupperまでの一様分布）
    c22 = pymc.Uniform('c22', lower=0, upper=100000)#c22の初期分布（lowerからupperまでの一様分布）
    c30 = pymc.Uniform('c30', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    c31 = pymc.Uniform('c31', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    c32 = pymc.Uniform('c32', lower=0, upper=100000)#c32の初期分布（lowerからupperまでの一様分布）
    c40 = pymc.Uniform('c40', lower=0, upper=10000000)#c41の初期分布（lowerからupperまでの一様分布）
    c41 = pymc.Uniform('c41', lower=0, upper=10000000)#c41の初期分布（lowerからupperまでの一様分布）
    c42 = pymc.Uniform('c42', lower=0, upper=10000000)#c42の初期分布（lowerからupperまでの一様分布）
    c50 = pymc.Uniform('c50', lower=0, upper=100000)#c51の初期分布（lowerからupperまでの一様分布）
    c51 = pymc.Uniform('c51', lower=0, upper=100000)#c51の初期分布（lowerからupperまでの一様分布）
    c52 = pymc.Uniform('c52', lower=0, upper=100000)#c52の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def function(xp=xp, c10=c10, c20=c20, c30=c30, c40=c40, c50=c50, 
                 xm=xm, c11=c11, c21=c21, c31=c31, c41=c41, c51=c51, 
                 c12=c12, c22=c22, c32=c32, c42=c42, c52=c52):
        t1 = ( c10 + c11*np.power(xm,3) + c12*np.power(xm,2) ) / xp
        t2 = ( c20 + c21*np.power(xm,3) + c22*np.power(xm,2) )
        t3 = ( c30 + c31*xm + c32 ) * np.log(xp)
        t4 = ( c40 + c41*np.power(xm,3) + c42*np.power(xm,2) ) / np.power(xp,2)
        t5 = ( c50 + c51*np.power(xm,2) + c52*xm ) * np.log(xp) / np.sqrt(xp)
        return(t1+t2+t3+t4+t5)
        
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()



def test_model_253(xp, xm, y): # 簡易モデル
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=1000)#c21の初期分布（lowerからupperまでの一様分布）
    # c3 = pymc.Uniform('c30', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def function(xp=xp, xm=xm, c1=c1, c2=c2):
        return( (c1*np.power(xm,2)) / xp + (c2*np.power(xm,2)) * np.log(xp) )
        
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()

def test_model_253l(xp, xm, y): # 簡易モデル
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=1000)#c21の初期分布（lowerからupperまでの一様分布）
    # c3 = pymc.Uniform('c30', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def function(xp=xp, xm=xm, c1=c1, c2=c2):
        lis = list(map(lambda ep,em :((c1*np.power(em,2)) / ep + (c2*np.power(em,2)) * np.log(ep)), xp,xm ))
        return(lis)
        
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()

def test_model_253lb(x, y): # 簡易モデル xはxp,xmのStackしたもの
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=1000)#c21の初期分布（lowerからupperまでの一様分布）
    # c3 = pymc.Uniform('c30', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    #eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.00001)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def function(x=x, c1=c1, c2=c2):
        xp=list(x[:,0])
        xm=list(x[:,1])
        lis = list(map(lambda ep,em :((c1*np.power(em,2)) / ep + (c2*np.power(em,2)) * np.log(ep)), xp,xm ))
        return(lis)
        
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()



def test_model_254(xp, xm, y): # 簡易モデル
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=1000)#c21の初期分布（lowerからupperまでの一様分布）
    # c3 = pymc.Uniform('c30', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def function(xp=xp, xm=xm, c1=c1, c2=c2):
        return( (c1*np.power(xm,2)) / xp + (c2*np.power(xm,2)) * np.log(xp) )
        #lis = list(map(lambda ep,em :((c1*np.power(em,2)) / ep + (c2*np.power(em,2)) * np.log(ep)), xp,xm ))
        #return(lis)
        
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()

#def test_model_255(xpl, xml, yl): # T(P,M)簡易モデル LOG版
#    import pymc
#    import numpy as np
#    
#    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c11の初期分布（lowerからupperまでの一様分布）
#    c2 = pymc.Uniform('c2', lower=0, upper=1000)#c21の初期分布（lowerからupperまでの一様分布）
#    # c3 = pymc.Uniform('c30', lower=0, upper=100000)#c31の初期分布（lowerからupperまでの一様分布）
#    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
#    
#    @pymc.deterministic 
#    def function(xpl=xpl, xml=xml, c1=c1, c2=c2):
#        #return( (c1*np.power(xm,2)) / xp + (c2*np.power(xm,2)) * np.log(xp) )
#        return np.log( c1*np.exp(xml*3-xpl) + c2*xpl*np.exp(xml*2) )
#        #lis = list(map(lambda ep,em :((c1*np.power(em,2)) / ep + (c2*np.power(em,2)) * np.log(ep)), xp,xm ))
#        #return(lis)
#        
#    @pymc.deterministic
#    def tau(eps=eps):
#        return np.power(eps, -2)
#    
#    yl = pymc.Normal('yl', mu=function, tau=tau, value=yl, observed=True)
#    return locals()

def LogT_cost_255(xpl, xml, c1, c2):
    import numpy as np
    return( np.log( c1*np.exp(xml*3-xpl) + c2*xpl*np.exp(xml*2) )) # PyMC非依存なので使いまわせる関数
    
    # # Usage 1 # # 
    #@pymc.deterministic 
    #def model(xpl=xp_sample_l, xml=xm_sample_l, c1=c1, c2=c2):
    #    return LogT_cost_255(xpl, xml, c1, c2)
    #
    # # Usage 2 # # 
    #from models_for_BayesianPerformancePredictionTools import LogT_cost_255
    #Log_Time = LogT_cost_255(xp_sample_l, xm_sample_l, c1, c2)
    # 
    
def test_model_255(xp_sample_l, xm_sample_l, y_sample_l): # T(P,M)簡易モデル LOG版　数値はNumpy.arrayで渡してください
    import pymc
    import numpy as np
    from models_for_BayesianPerformancePredictionTools import LogT_cost_255
    
    c1 = pymc.Uniform('c1', lower=0, upper=1000, value=500 ) # c11の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=0, upper=100000, value=50000 ) # c21の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.5) # 誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
    
    @pymc.deterministic 
    def model(xpl=xp_sample_l, xml=xm_sample_l, c1=c1, c2=c2):
        return LogT_cost_255(xpl, xml, c1, c2)
    
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    return_from_model = pymc.Normal('return_from_model', mu=model, tau=tau, value=y_sample_l, observed=True)
    return locals()
    ##Usage
    #mcmc = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l))
    #mcmc.sample(iter=200000, burn=100, thin=10)

def test_model_131(x, y, l):
    import pymc
    import numpy as np
    
    c1 = pymc.Uniform('c1', lower=-100000, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
    c2 = pymc.Uniform('c2', lower=-100000, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
    c3 = pymc.Uniform('c3', lower=-100000, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
    eps = pymc.Uniform('eps', lower=0, upper=0.00001)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）

    @pymc.deterministic
    def function(x=x, c1=c1, c2=c2, c3=c3):
        if l:
            return (c1 / np.exp(x)) + c2 + (c3 * x)
        else:
            return (c1 / np.exp(x)) + c2 + (c3 * np.exp(x))
    
    
    @pymc.deterministic
    def tau(eps=eps):
        return np.power(eps, -2)
    
    y = pymc.Normal('y', mu=function, tau=tau, value=y, observed=True)
    return locals()





# old #-----------------------------------------------------

## 使用禁止
#def test_model_152(x, y):
#    c1 = pymc.Uniform('c1', lower=0, upper=100000)#c1の初期分布（lowerからupperまでの一様分布）
#    c2 = pymc.Uniform('c2', lower=0, upper=100000)#c2の初期分布（lowerからupperまでの一様分布）
#    c3 = pymc.Uniform('c3', lower=0, upper=100000)#c3の初期分布（lowerからupperまでの一様分布）
#    c4 = pymc.Uniform('c4', lower=0, upper=10000000)#c4の初期分布（lowerからupperまでの一様分布）
#    c5 = pymc.Uniform('c5', lower=0, upper=100000)#c5の初期分布（lowerからupperまでの一様分布）
#    eps = pymc.Uniform('eps', lower=0, upper=0.5)#誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
#    
#    #@pymc.deterministic
#    #def function(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5):
#    #    return np.log((c4 / np.exp(2.0*x)) + (c1 / np.exp(x)) + c2 + (c3 * np.exp(x)) + c5*(np.log(x)/np.sqrt(x)))
#    
#    @pymc.deterministic
#    def function_l(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l:
#        obj = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + (c5 * elem)/(np.exp(0.5*elem))), x ))
#        return obj
#    
#    @pymc.deterministic
#    def function_l100(x=x, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5): # if l: if x[i]>100:
#        term5 = 0.0 # term5 = (c5 * x[i])/(np.exp(0.5*x[i]))
#        obj = list(map(lambda elem : np.log((c4 / np.exp(2.0*elem)) + (c1 / np.exp(elem)) + c2 + (c3 * elem) + term5), x ))
#        return obj
#    
#    @pymc.deterministic
#    def tau(eps=eps):
#        return np.power(eps, -2)
#    
#    y = pymc.Normal('y', mu=function_l, tau=tau, value=y, observed=True)
#    return locals()
#
##model = pymc.Model([function_l, x])
#pymc.numpy.random.seed(0)
#mcmc = pymc.MCMC(test_model_152(x_sample_l, y_sample))
#mcmc.sample(iter=100000, burn=50000, thin=10)

#def test_model_255(xp_sample_l, xm_sample_l, y_sample_l): # T(P,M)簡易モデル LOG版　数値はNumpy.arrayで渡してください
#    import pymc
#    import numpy as np
#    import LogT_cost_255
#    
#    c1 = pymc.Uniform('c1', lower=0, upper=100000) # c11の初期分布（lowerからupperまでの一様分布）
#    c2 = pymc.Uniform('c2', lower=0, upper=1000) # c21の初期分布（lowerからupperまでの一様分布）
#    eps = pymc.Uniform('eps', lower=0, upper=0.2) # 誤差パラメータepsの初期分布（lowerからupperまでの一様分布）
#    
#    #@pymc.deterministic 
#    #def model(xpl=xp_sample_l, xml=xm_sample_l, c1=c1, c2=c2):
#    #    return np.log( c1*np.exp(xml*3-xpl) + c2*xpl*np.exp(xml*2) )
#    @pymc.deterministic 
#    def model(xpl=xp_sample_l, xml=xm_sample_l, c1=c1, c2=c2):
#        return LogT_cost_255(xpl, xml, c1, c2)
#    
#    @pymc.deterministic
#    def tau(eps=eps):
#        return np.power(eps, -2)
#    
#    return_from_model = pymc.Normal('return_from_model', mu=model, tau=tau, value=y_sample_l, observed=True)
#    return locals()
#    ##Usage
#    #mcmc = pymc.MCMC(test_model_255(xp_sample_l, xm_sample_l, y_sample_l))
#    #mcmc.sample(iter=200000, burn=100, thin=10)
    






