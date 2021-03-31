# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from RBM import RBM
from optimizer import Optimizer
from performance_metrics import Metrics_monitor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
from sklearn.neural_network import BernoulliRBM
from scipy.stats import pearsonr
from itertools import combinations
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import os
from statistics import mean, stdev
import openpyxl
import io
import tensorflow as tf

####################################################################
eps_min=0
eps_max=0

weeks=13
simlen=5


def data_fun():
    spx=pd.read_excel("SPX.xls", sheet_name="SPX", usecols=["Date","Adj Close"],header=0)
    spx.rename(columns={"Adj Close": "SP500"}, inplace=True)
    vix=pd.read_excel("VIX.xls", sheet_name="VIX", usecols=["Date","Adj Close"],header=0)
    vix.rename(columns={"Adj Close": "VIX"}, inplace=True)
    spx.Date=pd.to_datetime(spx.Date)
    spx.set_index('Date', inplace=True)
    vix.Date=pd.to_datetime(vix.Date)
    vix.set_index('Date', inplace=True)
    a=pd.merge(spx, vix, how="inner", right_index=True, left_index=True).sort_index()
    return a


def backfilling(data, logs=True):
    l = pd.date_range(data.index[0], data.index[-1], freq="B")
    dind = data.index
    new = pd.DataFrame(index=l, columns=data.columns)
    if logs:
        # backfill logs
        for i in range(len(l)):
            if l[i] in dind:
                new.loc[l[i], :] = data.loc[l[i]]
            else:
                new.loc[l[i], :] = np.zeros((1, 2))
    else:
        # backfill levels
        for i in range(len(l)):
            if l[i] in dind:
                new.loc[l[i], :] = data.loc[l[i]]
            else:
                new.loc[l[i], :] = new.loc[l[i - 1]]
    return new

def buildataset(data, freq="W"):
    windows=[]
    for _, w in data.resample(freq):
        windows.append(w)
    a=np.array([path for path in windows], dtype=object)
    return np.array([p for p in a if p.shape[0]==5], dtype=object)



def int_to_bin(data):
    #each column represents one ticker
    data=pd.DataFrame(data=data)
    num_var=data.shape[1]
    #each row contains hisorical forex values
    num_sample=data.shape[0]
    bin_dict={}
    x_bin=pd.DataFrame(index=data.index, columns=data.columns)
    x_min=pd.DataFrame(index=['min value'])
    x_max=pd.DataFrame(index=['max value'])
    for i in range(num_var):
        x_min.loc['min value',i]=data.iloc[:,i].min()-eps_min
        x_max.loc['max value',i]=data.iloc[:,i].max()+eps_max
        try:
            #print(65535*(data.iloc[:,i]-x_min.loc['min value',i])/(x_max.loc['max value',i]-x_min.loc['min value',i]))
            x_bin.iloc[:,i]=(65535*(data.iloc[:,i]-x_min.loc['min value',i])/(x_max.loc['max value',i]-x_min.loc['min value',i])).apply(lambda v: (format(int(v), '016b')))
            
            #gives (num_sample, num_var) shaped array such that each entry is a 16 digit binary number
        except ValueError:
            print("NaN value in calculation")
            return
    for i in list(x_bin):
        #convert each ticker separately into N_samp x 16 df and save separately in dictionary
        bin_dict[i]=x_bin[i].astype(str).apply(list).apply(pd.Series).astype(np.int64)
    
    #Num_samples x 64
    res=pd.concat(bin_dict.values(), ignore_index=True, axis=1)
    return res,x_max, x_min
        
def bin_to_int(bin_data):
    #, columns=['USDEUR.Curncy.Log.Returns','USDGBP.Curncy.Log.Returns','USDJPY.Curncy.Log.Returns','USDCAD.Curncy.Log.Returns'        
    partition=bin_data.shape[1]//16
    bindf=pd.DataFrame(data=bin_data)
    x_real=pd.DataFrame(index=bindf.index, columns=[i for i in range(partition)])

    x_int=np.zeros((bin_data.shape[0],partition))
    l=[[i*16, (i+1)*16] for i in range(partition)]
    
    for i in range(len(l)):
      
        temp=bin_data.iloc[:, l[i][0]:l[i][1]]
        
        for m in range(16):
            x_int[:,i] += (2**m)*temp.iloc[:,15-m]
            #xmin xmax are rows with each column being min,max of one currency exchange type
        x_real.iloc[:, i]=x_min.loc['min value',i]+x_int[:,i]*(x_max.loc['max value',i]-x_min.loc['min value',i])/65535
        #x_real is samplesize x 4, each column containing values of one ticker
    return x_real
    
    
def int_sample(inpt,vis_dim, hid_dim, num_epochs, MC_step,pic_shape, bat_size,  sample_size,zeit):
    
    machine=RBM(visible_dim=vis_dim,hidden_dim=hid_dim,number_of_epochs=num_epochs,picture_shape=pic_shape,batch_size=bat_size)
    machine.from_saved_model(r'results/models/' +zeit[0:4]+'/'+ zeit + 'model.h5')
    if len(inpt)==0:
        print("randomized input")
        s,_,i=machine.parallel_sample(n_step_MC=MC_step,n_chains=sample_size)
    else:
        s,_,i=machine.parallel_sample(inpt,n_ste_MC=MC_step,n_chains=sample_size)
    resdf=pd.DataFrame(data=s)
    # res_int shape: sample_size x 64
    res_int=bin_to_int(resdf)
    return res_int



def int_sample_fix(inp,vis_dim, hid_dim, num_epochs, pic_shape, bat_size,  sample_size,datei):
    
    machine=RBM(visible_dim=vis_dim,hidden_dim=hid_dim,number_of_epochs=num_epochs,picture_shape=pic_shape,batch_size=bat_size)
    
    machine.from_saved_model(r'results/models/' + datei[0:4] +'/' + datei)
    
    s,_,i=machine.parallel_sample(n_step_MC=1000,n_chains=sample_size)
    resdf=pd.DataFrame(data=s)
    # res_int shape: sample_size x 64
    res_int=bin_to_int(resdf)
    return res_int

def dataplot(data):
    n_sub=data.shape[1]
    f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, sharey=True)
    for i in range(n_sub):
        achse="ax"+str(i)
        ac=eval(achse)
        titel=data.columns[i][0:6]
        ac.hist(data.iloc[:,i])
        ac.set_title(titel)
       

#%%
    
if __name__ == '__main__':
    raw_data = data_fun()
    rawdata=backfilling(raw_data, logs=False)
    rawdata.SP500 = np.log((rawdata.SP500 / rawdata.SP500.shift(1)).astype(np.float64))
    #contains first nan
    spx=buildataset(rawdata.SP500)
    vix=buildataset(rawdata.VIX)

    final_rawdata=np.concatenate([spx[1:,:], vix[1:,:], vix[:-1,-1].reshape(-1,1)], axis=1)
    training_data, x_max, x_min=int_to_bin(final_rawdata)



    trainsize=training_data.shape[0]-100
    samplesize=training_data.shape[0]
    x_train = training_data.iloc[0:trainsize,:].to_numpy().astype(np.float64)
    x_test = training_data.iloc[trainsize:,:].to_numpy().astype(np.float64)
    data={"x_train":x_train,"y_train":[], "x_test":x_test, "y_test":[]}
    
   ###############################################################################################
    vis_dim=(16*(raw_data.shape[1]*simlen+1))
    #hids=[50, 100, 150]
    #l=[0.0001,0.001,0.01]
    hids=[50]
    l=[0.0001]
    mc=[500]
    starts = [4695]
    conditioning_week = int(np.floor((starts[0] - 1) / 5))
    sampsize = weeks
    samptimes = 200
    samps = np.zeros((simlen * weeks, samptimes))
    sampint = np.zeros((sampsize, 16 * simlen))

    for i in range(len(hids)):
        hid_dim = hids[i]
        for j in range(len(l)):
            l_rate = l[j]
            for k in range(len(mc)):

                #num_epochs=50000
                num_epochs=1
                pic_shape=(1,vis_dim)
                bat_size=50

                n_step_MC=mc[k]
                steps=5
                n_chains=1

                machine=RBM(vis_dim,hid_dim,num_epochs,pic_shape,bat_size, k=steps)
                print(hid_dim, l_rate, n_step_MC, machine._current_time, flush=True)
                optimus=Optimizer(machine,l_rate,opt='adam')
                machine.save_param(optimus)
                monitor=Metrics_monitor(machine, metrics = ['sq_error','DKL'],steps=n_step_MC,n_chains=n_chains)
                #rawdata.sp500 contains first nan
                machine.train(data,rawdata.SP500.values[1:] ,optimus, monitor ,x_max,x_min)



                # inpt = np.random.choice([0, 1], size=(n_chains, self._v_dim), p=[p_0, p_1]).astype(np.float64)
                for i in range(samptimes):
                    print("samptime ", i, flush=True)
                    cond = data['x_train'][0, -16:]
                    for j in range(sampsize):
                        rnd = np.random.choice([0, 1], size=(1, 16*2*simlen), p=[0.5, 0.5]).astype(np.float64)
                        inpt = np.concatenate([rnd, cond.reshape(1, 16)], axis=1)
                        a, _, _ = machine.parallel_sample_cond(cond=cond.reshape(1, -1), inpt=inpt, n_step_MC=n_step_MC, n_chains=1)
                        # print(a)
                        sampint[j, :] = a[:, :16*simlen]

                        cond = a[:, -32:-16]
                    s = pd.DataFrame(data=sampint)
                    # print(s)
                    # bin_to_int assumes each passed column to be one variable!
                    samps[:, i] = bin_to_int(s).values.reshape(-1, )

                sampsdf = pd.DataFrame(data=samps)
                #sampsdf.to_pickle("RBM" + str(hid_dim)+'_' + str(l_rate)+'_'+str(n_step_MC)+".pkl")
                #samps_np = np.asarray(samps).reshape(-1, samptimes)
