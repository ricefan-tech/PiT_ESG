#%% GENERAL STUFF
    # -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:47:50 2020

@author: t656703
"""
import pandas as pd
import numpy as np
from Bootstrap import HS
import matplotlib.pyplot as plt
from sGARCH_real import sgarch
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot_2samples
from matplotlib import rc
import market_generator
import random
import time
from sklearn import utils
from cvae import CVAE
#if old version is used
import pickle5 as pickle
rc('font',**{'family':'serif','serif':['Times'], 'size':15})

#spxpath=r"\\ubsprod.msad.ubs.net\userdata\t656703\home\Documents\R\GARCH\SPXVIX.xlsx"
#fsipath=r"\\UBSPROD.MSAD.UBS.NET\userdata\t656703\home\Documents\Python Scripts\MarketGenerator\fsi.xlsx"
spxpath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/SPX.xls"
#fsipath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/FSI.xls"
vixpath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/VIX.xls"

qqpath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/result plots/QQ/"
statspath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/result plots/Stats/"
perfpath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/result plots/PF performance/"
picklepath=r"/Users/rui/Documents/Rui/UBS/MA/Proper test code/result plots/Pickled samples/"

#amount of seeds
runs=1
#amount of paths per seed
path=500
#amount of simulated weeks, default 3mths simulation
weeks=13
#amount of lags to include in acf
lags=30
simlength=5
#look back period for strategy in weeks
lookback=52

##################################################################################################
################################################# DATA PREPARATIONS ##############################
##################################################################################################
def logdata():
    spx=pd.read_excel(spxpath, sheet_name="SPX", usecols=["Date","Adj Close"],header=0)
    spx.rename(columns={"Adj Close": "SP500"}, inplace=True)
    vix=pd.read_excel(vixpath, sheet_name="VIX", usecols=["Date","Adj Close"],header=0)
    vix.rename(columns={"Adj Close": "VIX"}, inplace=True)
    spx.Date=pd.to_datetime(spx.Date)
    spx.set_index('Date', inplace=True)
    vix.Date=pd.to_datetime(vix.Date)
    vix.set_index('Date', inplace=True)
    spx.loc[:,"SP500"]=np.log(spx.loc[:,"SP500"] / spx.loc[:,"SP500"].shift(1))
    vix.loc[:,"VIX"]=np.log(vix.loc[:,"VIX"] / vix.loc[:,"VIX"].shift(1))
    a=pd.merge(spx, vix, how="inner", right_index=True, left_index=True).sort_index()
    return a.iloc[1:,:]

def datas():
    #returns levels
    spx=pd.read_excel(spxpath, sheet_name="SPX", usecols=["Date","Adj Close"],header=0)
    spx.rename(columns={"Adj Close": "SP500"}, inplace=True)
    vix=pd.read_excel(vixpath, sheet_name="VIX", usecols=["Date","Adj Close"],header=0)
    vix.rename(columns={"Adj Close": "VIX"}, inplace=True)
    spx.Date=pd.to_datetime(spx.Date)
    spx.set_index('Date', inplace=True)
    vix.Date=pd.to_datetime(vix.Date)
    vix.set_index('Date', inplace=True)
    return pd.merge(spx, vix, how="inner", right_index=True, left_index=True).sort_index()
        
def backfilling(data, logs=True):
    l=pd.date_range(data.index[0], data.index[-1], freq="B")
    dind=data.index
    new=pd.DataFrame(index=l, columns=data.columns)
    if logs:
        #backfill logs
        for i in range(len(l)):
            if l[i] in dind:
                new.loc[l[i],:]=data.loc[l[i],:]
            else: 
                new.loc[l[i],:]=np.zeros((1,2))
    else:
        #backfill levels
        for i in range(len(l)):
            if l[i] in dind:
                new.loc[l[i],:]=data.loc[l[i],:]
            else: 
                new.loc[l[i],:]=new.loc[l[i-1],:]
    return new
    
def buildataset(data, freq="W"):
    windows=[]
    for _, w in data.resample(freq):
        windows.append(w)
    a=np.array([path for path in windows], dtype=object)
    return np.array([p for p in a if p.shape[0]==5], dtype=object)


##################################################################################################
################################################# STATISTICAL STUFF ##############################
##################################################################################################

def onedstats(realdata,d, starts, simtype=""):
    a=pd.DataFrame(columns=["Mean","Mean Std", "Std. Dev.","Std. Std.", "1st perc","1st perc Std", "99th perc", "99th perc Std", "IQR","IQR std", "skew", "skew std", "kurt", "kurt std"], index=np.arange(d.shape[-1]))
    b=pd.DataFrame(columns=["Mean", "Std. Dev.", "1st perc", "99th perc", "IQR", "skew", "kurt"], index=np.arange(d.shape[-1]))
    #3d means it is concatenated along third dimension of simulation runs
    print("stats: ", weeks)
    for i in range(d.shape[-1]):
        real=realdata.iloc[starts[i]-1:starts[i]-1+simlength*weeks,:].values.astype(np.float64)
        data=d[:,:,i]
        a["Mean"][i]=np.mean(data, axis=(0,1))
        a["Mean Std"][i]=np.std(np.mean(data,axis=0).reshape(1,-1), axis=1)
        a["Std. Dev."][i]=np.mean(np.std(data, axis=0).reshape(1,-1), axis=1)
        a["Std. Std."][i]=np.std(np.std(data,axis=0).reshape(1,-1), axis=1)
        a["1st perc"][i]=np.mean(np.percentile(data, 1, axis=0).reshape(1,-1), axis=1)
        a["1st perc Std"][i]=np.std(np.percentile(data, 1, axis=0).reshape(1,-1), axis=1)
        a["99th perc"][i]=np.mean(np.percentile(data, 99, axis=0).reshape(1,-1), axis=1)
        a["99th perc Std"][i]=np.std(np.percentile(data, 99, axis=0).reshape(1,-1), axis=1)
        a["IQR"][i]=np.mean((np.percentile(data, 75, axis=0)-np.percentile(data, 25, axis=0)).reshape(-1, data.shape[-1]), axis=1)
        a["IQR std"][i]=np.std((np.percentile(data, 75, axis=0)-np.percentile(data, 25, axis=0)).reshape(-1, data.shape[-1]), axis=1)
        a["skew"]=np.mean(st.skew(data, axis=0))
        a["skew std"]=np.std(st.skew(data, axis=0))
        a["kurt"]=np.mean(st.kurtosis(data, axis=0))
        a["kurt std"]=np.std(st.kurtosis(data, axis=0))

        
        
        b["Mean"][i]=np.mean(real, axis=0)
        b["Std. Dev."][i]=np.std(real, axis=0)
        b["1st perc"][i]=np.percentile(real, 1, axis=0)
        b["99th perc"][i]=np.percentile(real, 99, axis=0)
        b["IQR"][i]=np.percentile(real, 75, axis=0)-np.percentile(real, 25, axis=0)
        b["skew"]=st.skew(real, axis=0)
        b["kurt"]=st.kurtosis(real, axis=0)
        a.to_csv(statspath+simtype+str(starts[i])+".csv")
        b.to_csv(statspath+"real"+str(starts[i])+".csv")
    return a,b

#against real data
def qq2(r, data, starts, simtype="", save=True):
    real=np.asarray(r)
    for i in range(data.shape[-1]):
        rpart=real[starts[i]:starts[i]+weeks*simlength,0]
        data[:,:,i].sort(axis=0)
        dmean=np.mean(data[:,:,i], axis=1)
        print(np.mean(dmean))
        plt.figure()
        qqplot_2samples(rpart.reshape(1,-1), dmean.reshape(1,-1))
        plt.plot([-0.04, 0.04], [-0.04, 0.04], "black")
        plt.axis('square')
        plt.xlim(-0.04,0.04)
        plt.ylim(-0.04,0.04)
        if save:
            plt.savefig(qqpath+simtype+str(starts[i])+".svg")
    
#against fitted normal
def qqnorm(sims,simtype="", save=True):
    #takes in 2D simulation array
    sims.sort(axis=0)
    a=np.mean(sims, axis=1)
    print(np.mean(a))
    print(np.std(a))
    plt.figure()
    sm.qqplot(a, loc=np.mean(a), scale=np.std(a))
    plt.plot([-0.04, 0.04], [-0.04, 0.04], "black")
    plt.axis('square')
    plt.xlim(-0.04,0.04)
    plt.ylim(-0.04,0.04)
    plt.savefig(qqpath+simtype+".svg")
    
def acf(real, data, simtype=""):
    sim_acf=np.array([sm.tsa.acf(data[:,j], nlags=30) for j in range(data.shape[1])])
    print(sim_acf.shape)
    #real_acf=sm.tsa.acf(real.reshape(-1,1), nlags=30)
    real_acf=sm.tsa.acf(real, nlags=30)
    #sim_acf has shape sim_ret.shape[-2] x nlags+1 
    #res_acs=sim_acf.mean(axis=0)
    ax=plt.figure().gca()
    plt.plot(real_acf, label='Real Data', color="r")
    for j in range(3):
        i=np.random.randint(0, data.shape[1])
        plt.plot(sim_acf[i], label='Simulation '+str(j+1))
    plt.legend()
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    figure=plt.gcf()
    plt.tight_layout()
    plt.savefig(qqpath+simtype+"ACF.svg")
    plt.show()
    return sim_acf, real_acf

def acf_mean(real, data, simtype=""):
    
    ax=plt.figure().gca()
    plt.plot(real, label='Real Data', color="r")
    data_acf=np.array([sm.tsa.acf(data[:,j], nlags=30) for j in range(data.shape[1])])
    plt.plot(np.mean(data_acf, axis=0), label='Simulation Average')
    plt.legend()
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    figure=plt.gcf()
    plt.tight_layout()
    plt.savefig(qqpath+simtype+"averageACF.svg")
    plt.show()
    
    
def generate_from_saved(CVAE, cond, n_samples=None, inpt=5):
    cond = utils.as_float_array(cond)
    if n_samples is not None:
        randoms = np.random.normal(0, 1, size=(n_samples, CVAE.n_latent))
        cond = [list(cond)] * n_samples
    else:
        randoms = np.random.normal(0, 1, size=(1, CVAE.n_latent))
        cond = [list(cond)]

    samples = CVAE.decoder(randoms, cond, inpt)

    if n_samples is None:
        return samples[0]

    return samples
def from_pickle(name=""):
    r_rec=pd.read_pickle(name)
    with open(name, "rb") as fh:
        return pickle.load(fh)

##################################################################################################
################################################# STRATEGY FORECAST ##############################
##################################################################################################

def strat_dec(data, weeks, vol_limit, mean_limit,length=5, high_lim=0.5, low_lim=3):
    #get 2D data of simsxstarts
    res=np.zeros(shape=data.shape)
    for i in range(data.shape[-1]):
        for j in range(weeks):
            s=j*length
            e=(j+1)*length
            proj_mean=np.mean(data[s:e,i])
            proj_std=np.std(data[s:e,i])
            benchmark=mean_limit-stop_loss_factor(proj_std, vol_limit, high_lim, low_lim)*vol_limit
            if proj_mean>=benchmark:    
                res[s:e,i]= np.ones((length,))
            else:
                break
    return res

def stop_loss_factor(proj_std, vol_limit, high_lim=0.5, low_lim=3):
    return (proj_std>=vol_limit)*high_lim+(proj_std<vol_limit)*low_lim

def cum_ret(data):
    i=0
    while (data[i]!=0)&(data[i+1]!=0):
        res=(data[i]-data[0])/data[0]
        i+=1
        
        if i==data.shape[0]-1:
            break
    return res

def strategy_ret(final, simtype="", save=True):
    retpnl=[]
    for i in range(final.shape[-1]):
        if final[0,i]!=0:
            retpnl=retpnl+list(cum_ret(final[:,i:(i+1)]))
    print("Mean "+simtype+": ", np.mean(retpnl))
    if save:
        plt.figure()
        sns.histplot(retpnl, bins=50)
        plt.xlabel("Cumulative Returns")
        #plt.ylim(0,15)
        plt.xlim(-0.5,0.5)
        plt.ylim(0.0,30.0)
        plt.savefig(perfpath+simtype+"hist.svg")
        plt.show()
    return retpnl

def max_drawdown(simpf):
    #assume simpf to be simlength x simpfs
    drawdowns=[]
    for i in range(simpf.shape[-1]):
        X=list(simpf[:,i])
        if X[0]!=0:
            mdd = 0
            peak = X[0]
            for x in X:
                if x==0:
                    break
                if x > peak: 
                    peak = x
                dd = (peak - x) / peak
                if dd > mdd:
                    mdd = dd
            drawdowns.append(mdd)
        else:
            pass
    return drawdowns

def sim_acf(starts, real, sim_ret, simdays=weeks*5, simtype=""):
    z=sim_ret.shape[-1]
    for i in range(z):
        s=starts[i]        
        sim_acf=np.array([sm.tsa.acf(sim_ret[:,0,j,i], nlags=lags) for j in range(sim_ret.shape[-2])])
        res_acs=sim_acf.mean(axis=0)
        plt.figure()
        plt.plot(res_acs, label='Generated Data')
        plt.plot(sm.tsa.acf(cum_ret(real[s*5:s*5 + simdays, 0]), nlags=lags), label='Real Data')
        plt.legend()
        plt.xlabel("Lags")
        plt.ylabel("Autocorrelation")
        figure=plt.gcf()
        figure.set_size_inches(5, 3)
        plt.tight_layout()
        plt.savefig(perfpath+simtype+"/"+"ACF"+str(starts[i]))
        plt.show()    

import pickle5 as pickle

def from_pickle(name=""):
    r_rec=pd.read_pickle(name)
    with open(name, "rb") as fh:
        return pickle.load(fh)

def save_plot(data,gar,fhs,cvae,simtype=""):
    plt.figure()
    sns.histplot(data, bins=50)
    plt.xlabel("Cumulative Returns")
    #plt.ylim(0,15)
    plt.xlim(-0.2,0.3 )
    plt.savefig(perfpath+simtype+"hist.svg")
    plt.show()
    
#%% MAIN

if __name__ == '__main__':
    real_level=backfilling(datas(), logs=False)
    
    raw=datas()
    data=backfilling(raw)
    runs=1
    time_index=data.index
    datas_np=np.asarray(data)
    
    np.random.seed(1)
    dataset=buildataset(data)
    #starts samples number of random week, datas is daily data
    starts = np.random.randint(100, dataset.shape[0], runs)
    #starts=starts[1:]
    starts=[4695]
    real_ret=backfilling(logdata())

    real_acf=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values, nlags=30)
    real_acf_sq=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, nlags=30)
    #%%
    #testing 2019 april first
    starts=[5085]
    ##################################################################################################
    fig, ax1=plt.subplots()
    ax1.set_ylabel("SPX Log Returns")
    ax1.plot(dfil.SP500.iloc[4695:4695+65], color="g", label='SPX')
    ax2=ax1.twinx()
    ax2.set_ylabel("VIX Index")
    ax2.plot(dfil.VIX.iloc[4694:4694+65], color="b", label="VIX")
    fig.autofmt_xdate()
    lines= ax1.get_lines() + ax2.get_lines()
    plt.legend(lines, [l.get_label() for l in lines], loc="lower right")
    plt.savefig(statspath+"spxvix.svg")
#%% 
##################################################################################################
################################################# STATISTICAL STUFF ##############################
##################################################################################################

    #%% FHS STATS
    
    ##################################################################################################
    ################################################# FHS ############################################
    ##################################################################################################
        
    d=datas()
    #dfil contains first NaN
    runs=len(starts)
    
    w= [4, 26, 52]
    
    for i in range(len(w)):
        weeks=w[i]
        dfil=backfilling(d, logs=False)
        bs=HS(dfil, weeks, path)
        sim_ret=np.zeros((weeks*simlength,path, runs))
        #choose starts0-1 because it needs to be last day before training period VIX 
        sim_ret[:,:,0]=bs.BS(dfil.iloc[starts[0]-1:starts[0]-1+simlength*weeks,1].values.reshape(-1,1))
    
        simretdf=pd.DataFrame(data=sim_ret[:,:,0])
        simretdf.to_pickle(picklepath+str(weeks)+"_BSsamps010419.pkl")
        #sim_ret[:,:,0]=np.asarray(from_pickle(picklepath+str(weeks)+"_BSsamps010118.pkl"))
        
        real_ret=backfilling(logdata())
        #if starts is 4695
        real_acf=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values, nlags=30)
        real_acf_sq=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, nlags=30)
        acf_mean(real_acf,sim_ret[:,:,0], simtype=str(weeks)+"BS_nonsq_new")
        acf_mean(real_acf_sq, sim_ret[:,:,0]**2, simtype=str(weeks)+"BS_new")
        ### qqplot real return vs simulated returns & fitted gaussian
        qq2(real_ret, sim_ret[:,:,0:1], starts,  simtype=str(weeks)+"BS_new")
        qqnorm(sim_ret[:,:,0], simtype=str(weeks)+"BS_new")
        ### compare stats
        a,b=onedstats(real_ret.iloc[:,0:1],sim_ret, starts, simtype=str(weeks)+"BS_new")
  
    
    #%% SGARCH STATS

    ##################################################################################################
    ################################################# sGARCH ########################################
    ##################################################################################################
    
    gar=sgarch(dist="t")
    d=datas()
    #dfil contains first NaN
    dfil=backfilling(d, logs=False)
    #dfil[['SP500']]=dfil[['SP500']]/dfil[['SP500']].shift(1)


    dfil[['SP500']]=np.log((dfil[['SP500']]/dfil[['SP500']].shift(1)).astype(np.float64))
    #dfil[['VIX']]=np.log((dfil[['VIX']]/dfil[['VIX']].shift(1)).astype(np.float64))
   
    
    i=0
    w= [4, 26, 52]
    
    
    while i!=runs:
        try:
            gar.garch_fit(dfil.iloc[1:,:].values)
            #dfil needs index 2694 for 2017 12 31
            volstart=(1/gar.gamma)*(1/22*(dfil.iloc[starts[0],1]/100)**2-gar.sigma_bar*(1-gar.gamma))
            if volstart<0:
                print("starting vol <0")
                break
            for j in range(3):
                weeks=w[j]
                sims_ret_gar=np.zeros((weeks*simlength,path, runs))
                
                _,sims_ret_gar[:,:,0],sigs=gar.generate(startvol=volstart,starts=starts[0]-1, num_path=path, path_length=weeks*simlength)
                #_,sims_ret_gar[:,:,i],sigs=gar.generate(startret=dfil.iloc[starts[0]-1,0],startvol=gar.sigma2,starts=starts[0]-1, num_path=path, path_length=weeks*simlength)
                s_ret=pd.DataFrame(data=sims_ret_gar[:,:,0])
                s_ret.to_pickle(picklepath+str(weeks)+"_GARsamps010118_new.pkl")
                real_ret=backfilling(logdata())
                #sims_ret_gar[:,:,0]=np.asarray(from_pickle(picklepath+str(weeks)+"_GARsamps010118.pkl"))
                real_acf=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values, nlags=30)
                real_acf_sq=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, nlags=30)
                acf_mean(real_acf, sims_ret_gar[:,:,0], simtype=str(weeks)+"GAR_nonsq_new")  
                acf_mean(real_acf_sq,sims_ret_gar[:,:,0]**2, simtype=str(weeks)+"GARreal_new")  
                #stats on returns#
                
                simstats, realstats=onedstats(realdata=real_ret[["SP500"]],d=sims_ret_gar, starts=starts, simtype=str(weeks)+"GAR_new")
                #qq plots on returns 
                qq2(real_ret, sims_ret_gar[:,:,0:1], [starts[0]-1], simtype=str(weeks)+"GAR_new")
                qqnorm(sims_ret_gar[:,:,0], simtype=str(weeks)+"GAR_new")
                i+=1
        except OverflowError:
            pass
    
    
  
    
    #%% CVAE STATS
    ##################################################################################################
    ################################################# CVAE ###########################################
    ##################################################################################################
    
    ticker = "^GSPC"
    ticker2="^VIX"
    MG = market_generator.MarketGenerator(ticker, ticker2,nhidden=30, nlatent=4, alpha=0.001,
                                          freq="W", sig_order=None)
    
    #MG.train(n_epochs=20500, lrate= 0.0005)
    MG.train(n_epochs=20500, lrate= 0.0005)
    real_ret=backfilling(logdata())
    

    conditioning_week=int(np.floor((starts[0]-1)/5))
    #need conditioning week because need to start from week before 
    #sim_cvae=MG.sim(MG.conditions[conditioning_week], weeks, path)
    w= [4, 26, 52]
    for i in range(3):
        weeks=w[i]
        sim_cvae=MG.sim2(MG.conditions[conditioning_week:conditioning_week+weeks], weeks, path)
        cond_week_uncut=int(np.floor((starts[0])/5))
       
        #reshape simulation array 
        sim_cvae_twod=np.zeros((weeks*simlength,path,runs))
        for i in range(path):
            sim_cvae_twod[:,i]=sim_cvae[:,:,i].reshape(-1,1)
        
        sret=pd.DataFrame(data= sim_cvae_twod[:,:,0])
        #sret.to_pickle(picklepath+str(weeks)+"_CVAEstatssamps.pkl")
        #sim_cvae_twod[:,:,0]=np.asarray(from_pickle(picklepath+str(weeks)+"_CVAEstatssamps.pkl"))
        
        real_ret=backfilling(logdata())
        
        #starts0 minus one if starts0 is 4695
        cvae_stats, realstats=onedstats(real_ret[["SP500"]], sim_cvae_twod, [starts[0]-1])
        #acf(MG.orig_logsig[cond_week_uncut:cond_week_uncut+simlength*weeks,:].reshape(-1,1) ,sim_cvae_twod[:,:,0], simtype=str(weeks)+"CVAE_nonsq")
        #cvae_acf, real_acf=acf_mean(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values,sim_cvae_twod[:,:,0], simtype=str(weeks)+"CVAE_nonsqtest")
        #cvae_acf_sq, real_acf_sq=acf_mean(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, sim_cvae_twod[:,:,0]**2, simtype=str(weeks)+"CVAErealtest")   
        real_acf=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values, nlags=30)
        real_acf_sq=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, nlags=30)
        
        acf_mean(real_acf,sim_cvae_twod[:,:,0], simtype=str(weeks)+"CVAE_nonsq")
        acf_mean(real_acf_sq, sim_cvae_twod[:,:,0]**2, simtype=str(weeks)+"CVAEreal")
        qq2(real_ret,sim_cvae_twod[:,:,0:1], [starts[0]-1], simtype=str(weeks)+"CVAE" )
        qqnorm(sim_cvae_twod[:,:,0], simtype=str(weeks)+"CVAE")
  
    #%%
    
    load_cvae=CVAE(n_hidden=30, n_latent=4, alpha=0.001)
    load_cvae.model_from_saved(model_path="CVAE/models/1203", meta_path="CVAE/models/1203/1203-1527091000.meta")
    MG_new=market_generator.MarketGenerator(ticker, ticker2,nhidden=30, nlatent=4, alpha=0.001,
                                          freq="W", sig_order=None, saved_model=load_cvae)
    a=MG_new.sim(MG.conditions[conditioning_week:conditioning_week+weeks], weeks, path, saved_model=True, scaler_path1="CVAE/scalers/1203/1203-153134_scaler_data.save", scaler_path2="CVAE/scalers/1203/1203-153134_scaler_cond.save")
    #%%
    
        
      #%% RBM STATS
    ##################################################################################################
    ################################################# RBM ###########################################
    ##################################################################################################

    rbm1=from_pickle('RBM1302-165413_epoch30000model.pkl')
    sims_ret_rbm1=np.zeros((65,500,1))
    sims_ret_rbm1[:,:,0]=rbm1
    rbm_acf, real_acf=acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values, sims_ret_rbm1[:,:,0], simtype=str(weeks)+"RBM_nonsq_test")  
    rbm_acf_sq, real_acf_sq=acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, sims_ret_rbm1[:,:,0]**2, simtype=str(weeks)+"RBMreal_test")  
    #stats on returns
    simstats, realstats=onedstats(realdata=real_ret[["SP500"]],d=sims_ret_rbm1, starts=starts, simtype=str(weeks)+"RBM_test")
    #qq plots on returns 
    qq2(real_ret, sims_ret_rbm1[:,:,0:1], [starts[0]-1], simtype=str(weeks)+"RBM_test")
    qqnorm(sims_ret_rbm1[:,:,0], simtype=str(weeks)+"RBM_test")
    
    #%%
    w=[4,26,52]
    for i in range(3):
        weeks=w[i]
        file="RBM"+str(weeks)+".pkl"
        rbm=from_pickle(file)
        sims_ret_rbm=np.zeros((weeks*simlength,500,1))
        sims_ret_rbm[:,:,0]=rbm
        real_ret=backfilling(logdata())
        real_acf=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values, nlags=30)
        real_acf_sq=sm.tsa.acf(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values**2, nlags=30)
        
        acf_mean(real_acf,sims_ret_rbm[:,:,0], simtype=str(weeks)+"RBM_nonsq")
        acf_mean(real_acf_sq, sims_ret_rbm[:,:,0]**2, simtype=str(weeks)+"RBMreal")
        qq2(real_ret,sims_ret_rbm[:,:,0:1], [starts[0]-1], simtype=str(weeks)+"RBM" )
        qqnorm(sims_ret_rbm[:,:,0], simtype=str(weeks)+"RBM")
        
    #%%
    w=[4,26,52]
    for i in range(3):
        weeks=w[i]
        real_ret=backfilling(logdata())
        qqnorm(real_ret.iloc[starts[0]-1:starts[0]-1+simlength*weeks,0].values.reshape(-1,1), simtype=str(weeks)+"real")
    #%%
##################################################################################################
################################################# STRATEGY FORECAST ##############################

#%% 
    ##################################################################################################
    ################################################# PREPARATION ####################################
    ##################################################################################################
    
    simlength=5
    weeks=13
    random.seed(1)
    runs=200
    d=datas()
    #dfil contains first NaN
    dfil=backfilling(d, logs=False)
    sampleddata= buildataset(dfil)
    #foirst sample dates from 60th week (2001-feb-26) up tp excluding 2007-jan-01
    starts1=random.sample(range(60, 365), 50)
    #from 2007-jan-01 up to excluding 2010-jan-04
    starts2=random.sample(range(365, 522), 100)
    #exclude 5 weeks in the end to be sure
    starts3=random.sample(range(522, sampleddata.shape[0]-weeks-1), 50)
    starts = starts1+starts2+starts3
    #daily starts for data with length 547, i.e. uncut
    daily_starts=[a*5 for a in starts]
    #logdata returns without first value, ie. lags one value behind starts
    real_ret=backfilling(logdata())
    
#%% 
    ##################################################################################################
    ################################################# FHS ############################################
    ##################################################################################################
    random.seed() 
    d=datas()
    #dfil contains first NaN
    dfil=backfilling(d, logs=False)
    #tp store simulated portfolio evolution
    sim_ret_fhs=np.zeros((weeks*simlength, runs))
    for i in range(runs): 
        #+1 because HS calculates logs and then removes first nan
        #to run straetegy for 13 weeks need one more week to decide on stratagy in last week
        bs=HS(dfil.iloc[:daily_starts[i],:], weeks, 1)
        mean_limit=np.mean(real_ret.iloc[daily_starts[i]-lookback*simlength:daily_starts[i],0]) 
        vol_limit=np.std(real_ret.iloc[daily_starts[i]-lookback*simlength:daily_starts[i],0])
        #-1 because need last day vix before simulation start
        tmp=dfil.iloc[daily_starts[i]:daily_starts[i]+simlength*(weeks),1].values.reshape(-1,1)
        startvix=np.asarray([tmp[::5]]*5).T.reshape(-1,1)
        tmp=strat_dec(bs.BS(startvix), weeks, vol_limit, mean_limit)
        #use dfil ok if there are no logreturns
        sim_ret_fhs[:,i:(i+1)]=np.multiply(tmp, np.asarray(dfil.iloc[daily_starts[i]:daily_starts[i]+(weeks)*simlength,0:1]))
    savedf=pd.DataFrame(data=sim_ret_fhs)
    savedf.to_pickle(perfpath+str(weeks)+"_BSpfret_long.pkl")
    sim_ret_fhs=np.asarray(from_pickle(perfpath+str(weeks)+"_BSpfret_long.pkl"))
    fhs_mdd=max_drawdown(sim_ret_fhs)
    pnl_fhs=strategy_ret(sim_ret_fhs, simtype="BStest_long")
    
    
     #%%
    ##################################################################################################
    ################################################# GARCH ############################################
    ##################################################################################################
    
    gar=sgarch(dist="t")
    d=datas()
    #d####DFIL ALWYS CONTAINS FIRST NAN
    dfil=backfilling(d, logs=False)

    dfil[['SP500']]=np.log((dfil[['SP500']]/dfil[['SP500']].shift(1)).astype(np.float64))
    sim_ret_gar=np.zeros((weeks*simlength, runs))
    dfil2=backfilling(d, logs=False)
    count=0
    i=0
    j=0
    while i!=runs:
        print(i)
        tmp=np.zeros((weeks*simlength, 1))
        try:
            gar.garch_fit(dfil.iloc[1:daily_starts[i],:].values)
            mean_limit=np.mean(real_ret.iloc[daily_starts[i]-lookback*simlength:daily_starts[i],0]) 
            vol_limit=np.std(real_ret.iloc[daily_starts[i]-lookback*simlength:daily_starts[i],0])
            for j in range(weeks):
                s=j*simlength
                e=(j+1)*simlength
            
                if (1/gar.gamma)*(1/22*(dfil.iloc[daily_starts[i],1]/100)**2-gar.sigma_bar*(1-gar.gamma)) <0:
                    j+=1
                    pass
                #-2 from daily starts because gar.innovation starts one day later due to cut of log
                _,sims,_=gar.generate(startvol=(1/gar.gamma)*(1/22*(dfil.iloc[daily_starts[i],1]/100)**2-gar.sigma_bar*(1-gar.gamma)),starts=daily_starts[i]-2, num_path=1, path_length=simlength)
                tmp[s:e,:]=sims
            t=strat_dec(tmp, weeks,vol_limit, mean_limit)
            sim_ret_gar[:,i:(i+1)]=np.multiply(t, np.asarray(dfil2.iloc[daily_starts[i]:daily_starts[i]+weeks*simlength,0:1]))  
            print("i: {}, j: {}".format(i,j))
            i+=1
            j=0
        except OverflowError:
            pass
        
    savedf_gar=pd.DataFrame(data=sim_ret_gar)
    savedf_gar.to_pickle(perfpath+str(weeks)+"_GARpfret_long.pkl")
    sim_ret_gar=np.asarray(from_pickle(perfpath+str(weeks)+"_GARpfret_long.pkl"))
    gar_mdd=max_drawdown(sim_ret_gar)
    pnl_gar=strategy_ret(sim_ret_gar, simtype="GAR_long")
    
    
  
    #%%
    ##################################################################################################
    ################################################# CVAE ############################################
    ##################################################################################################
   
    ticker = "^GSPC"
    ticker2="^VIX"
   
    
    
    path=1
    sim_ret_cvae=np.zeros((weeks*simlength, runs))
    start=time.time()

    for i in range(runs):
        print(i)
        tmp=np.zeros((weeks*simlength, 1))
        mean_limit=np.mean(real_ret.iloc[daily_starts[i]-lookback*simlength:daily_starts[i],0]) 
        vol_limit=np.std(real_ret.iloc[daily_starts[i]-lookback*simlength:daily_starts[i],0])
        MG = market_generator.MarketGenerator(ticker, ticker2, end=dfil.index[daily_starts[i]] ,nhidden=30, nlatent=4, alpha=0.001,
                                         freq="W", sig_order=None)
        MG.train(n_epochs=20000, lrate= 0.0005, show=False)
        vols=dfil.iloc[daily_starts[i]:daily_starts[i]+simlength*weeks,1].values.reshape(-1,5)
        vol_conds=MG.scaler2.transform(vols)
        
        sims=MG.sim([MG.rounded_cond[-1,-1]]+ list(vol_conds[:,-1]),weeks, path)
        #strat dec takes as input weeks*simlength x path array
        tmp=strat_dec(sims[:,:,path-1].reshape(-1,1),weeks, vol_limit, mean_limit)
        sim_ret_cvae[:,i:(i+1)]=np.multiply(tmp, np.asarray(dfil2.iloc[daily_starts[i]:daily_starts[i]+weeks*simlength,0:1]))  
    end=time.time()
    dura=end-start
    # print("duration: ", dura)
    # for i in range(200):
    #     sim_ret_cvae[:,i:(i+1)]=np.multiply(tmp[:,i:(i+1)], np.asarray(dfil2.iloc[daily_starts[i]:daily_starts[i]+weeks*simlength,0:1]))
    savedf_gar=pd.DataFrame(data=sim_ret_cvae)
    savedf_gar.to_pickle(perfpath+str(weeks)+"_CVAEpfret.pkl")
    sim_ret_cvae=np.asarray(from_pickle(perfpath+str(weeks)+"_CVAEpfret.pkl"))
    cvae_mdd=max_drawdown(sim_ret_cvae)
    pnl_cvae=strategy_ret(sim_ret_cvae, simtype="CVAEPF")
    
    #%% PERFORMANCE BUY AND HOLD
    pnl=[]
    sim_ret_bh=np.zeros((weeks*simlength, runs))
    for i in range(runs):
        sim_ret_bh[:,i]=dfil2.iloc[daily_starts[i]:daily_starts[i]+weeks*simlength,0].values
        cumret=(dfil2.iloc[daily_starts[i]+weeks*simlength,0]-dfil2.iloc[daily_starts[i],0]-1)/dfil2.iloc[daily_starts[i],0]-1
        pnl=pnl+[cumret]    
    bh_mdd=max_drawdown(sim_ret_bh)
    
    #%%% TEMP STUFF
    plt.style.use("seaborn-bright")
    plt.figure()
    plt.scatter(dfil.iloc[:-1,1], dfil.iloc[1:,0])
    plt.ylabel("SPX Log Returns")
    plt.xlabel("VIX Index")
    plt.savefig("Corr_test.svg")
        
    #%%
    
    sim_ret_rbm=np.asarray(from_pickle("result plots/PF performance/13RBM_simpf.pkl"))
    rbm_mdd=max_drawdown(sim_ret_rbm)
    pnl_rbm=strategy_ret(sim_ret_rbm, simtype="RBM_test")    
        
    #%%
    
    plt.figure()
    pnl_fhs.remove(pnl_fhs[2])
    sns.histplot(pnl_fhs, bins=50)
    plt.xlabel("Cumulative Returns")
      
    plt.xlim(-0.2,0.3)
    plt.ylim(0.0,21.0)
    plt.savefig(perfpath+"BStesthist.svg")
    
