# -*- coding: utf-8 -*-

"""
Created on Tue Nov 10 15:42:15 2020
@author: t656703
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
import os
from statistics import mean, stdev
import openpyxl
import scipy.optimize as opt
import random
import sys
import warnings
warnings.filterwarnings("ignore")
####################################################################

class HNGARCH():
    

    def garch_fit(self, data):
        data=np.asarray(data)
        np.random.seed()
        gamma=np.random.uniform(-0.5, 10, 1)
        beta0=np.random.uniform(0, 10,1)
        beta3=np.random.uniform(-10,10,1)
        boundbeta2=min(1/beta3**2, 1/(beta3+gamma+0.5)**2)

        beta2=np.random.uniform(0,boundbeta2,1)
        #print("beta2", beta2)
        upper_bound= min((1-beta2*(beta3+gamma+0.5)**2), (1-beta2*beta3**2))
        beta1=np.random.uniform(0, upper_bound,1)
        #print(beta1)
        #init_val=(-0.1, 0.3, 0.5,0.4,0.4)
        init_val=(gamma,beta0,beta1,beta2,beta3)
        print(init_val)
        res=opt.minimize(self.LLH, x0=init_val, method="Nelder-Mead",args=(data), options={'maxiter': 1000, 'disp': True})
        self.gamma=res.x[0]
        self.beta0=res.x[1]
        self.beta1=res.x[2]
        self.beta2=res.x[3]
        self.beta3=res.x[4]
        

    
    def garchrec(self,params, data,r=0):
        #print("sigma")
        gamma=params[0]
        beta0=params[1]
        beta1=params[2]
        beta2=params[3]
        beta3=params[4]
        
        n=data.shape[0]
        sigma2=np.zeros(n)
        inno=np.zeros(n)
        
        Flag=False
        
        psi=beta1+beta2*beta3**2
        psi_tilde=beta1+beta2*(beta3+gamma+0.5)**2
        
        if (beta0<=0)| (min(beta1,beta2)<0) | (psi_tilde >=1) | (gamma <= -0.5):
            Flag=True
            return Flag, sigma2, inno
        
        for t in range(n):
            if t==0:
                sigma2[t]=np.var(data)
                #(beta0+beta1)/(1-beta1-beta2*beta3**2)
                #print(sigma2[t])
                inno[t]=(data[t]-r-gamma*sigma2[t])/np.sqrt(sigma2[t])
            else:
                sigma2[t]=beta0+beta1*sigma2[t-1]+beta2*(inno[t-1]-beta3*np.sqrt(sigma2[t-1]))**2
                inno[t]=(data[t]-r-gamma*sigma2[t])/np.sqrt(sigma2[t])
            if (sigma2[t]<0):
                print("neg sigma ", t)
                Flag=True
                return Flag, sigma2, inno
        return Flag, sigma2, inno
    
        
    def LLH(self,params, inpt, rho=0.5, r=0):#
        data=inpt[:,0]
        cond=inpt[:,1]
        gamma=params[0]
        beta0=params[1]
        beta1=params[2]
        beta2=params[3]
        beta3=params[4]
        n=data.shape[0]
        flag, sigma2,i =self.garchrec(params, data,r)
        
        if flag:
            return np.inf
        self.sigma=sigma2
        
        
        
        llh_ret=-n/2*np.log(2*np.pi)-0.5*np.sum(np.log(sigma2)+(data-r-gamma*sigma2)**2/sigma2)
      
        #print("returns ", llh_ret)
        tau=22
        TT=252
        self.psi_tilde=beta1+beta2*(beta3+gamma+0.5)**2
        self.h_bar=(beta0+beta2)/(1-self.psi_tilde)
        self.help_var=(1-self.psi_tilde**n)/((1-self.psi_tilde)*TT)
        vix_model=sigma2[1:]*self.help_var+self.h_bar*(1-self.help_var)
        vix_model=np.sqrt(vix_model*tau)*100        
        #ix model only has time index 0,...,n-1; bec it needs vola from next time step, hence need to take out last value from VIX as well
        VIX=cond[:-1]
        u=VIX-vix_model
        
        ########checking equation 10 validity######################
        spot=(1/self.help_var)*(1/22*(cond[:-1]/100)**2-self.h_bar*(1-self.help_var))
        if np.any(spot<0):
        #if spot[-1]<0:
            #print("negative spot")
            return np.inf
        #covariance of vector of observations is variance of vec
        sig=np.var(u)
        sig_prime=sig*(1-rho**2)
        llh_vix=-n/2*(np.log(2*np.pi)+np.log(sig_prime))+0.5*(np.log(sig_prime)-np.log(sig))-1/(2*sig)*(u[0]**2+np.sum((u[1:]-rho*u[:-1])**2/(1-rho**2)))
        #print("vix ", llh_vix)
        llh=llh_ret+llh_vix
        return -llh
    
    def generate(self, params, num_path, path_length, r=0):
        #generates given random params
        gamma=params[0]
        beta0=params[1]
        beta1=params[2]
        beta2=params[3]
        beta3=params[4]
        np.random.seed(1)
        sim_inno=np.random.normal(size=(path_length,num_path))
        sim_ret=np.zeros(shape=(path_length,num_path))
        sim_sig=np.zeros(shape=(path_length,num_path))
        for i in range(num_path):
            for j in range(1,path_length):
                sim_sig[j,i]=beta0+beta1*sim_sig[(j-1),i]+beta2*(sim_inno[(j-1),i]-beta3*np.sqrt(sim_sig[(j-1),i]))**2
                sim_ret[j,i]=r+gamma*sim_sig[j,i]+np.sqrt(sim_sig[j,i])*sim_inno[j,i]
        return sim_inno,sim_ret,sim_sig
    
   
    
    def sim(self, startret,startvix,num_path, path_length, r=0):
        np.random.seed()
        sim_inno=np.random.normal(size=(path_length,num_path))
        sim_ret=np.zeros(shape=(path_length,num_path))
        sim_sig=np.zeros(shape=(path_length,num_path))
        sim_sig[0,:]=startvix
        sim_ret[0,:]=startret
        for i in range(num_path):
            for j in range(1,path_length):
                sim_sig[j,i]=self.beta0+self.beta1*sim_sig[(j-1),i]+self.beta2*(sim_inno[(j-1),i]-self.beta3*np.sqrt(sim_sig[(j-1),i]))**2
                sim_ret[j,i]=r+self.gamma*sim_sig[j,i]+np.sqrt(sim_sig[j,i])*sim_inno[j,i]
    
    
        return sim_inno,sim_ret,sim_sig
    
    def generate_garch(self, params,  path_length,num_path=1, r=0):
        #generates given random params
        gamma=params[0]
        beta0=params[1]
        beta1=params[2]
        np.random.seed(1)
        sim_inno=np.random.normal(size=(path_length,num_path))
        sim_ret=np.zeros(shape=(path_length,num_path))
        sim_sig=np.zeros(shape=(path_length,num_path))
        for i in range(num_path):
            for j in range(1,path_length):
                sim_sig[j,i]=beta0+beta1*sim_sig[(j-1),i]+beta2*sim_ret[(j-1),i]**2
                sim_ret[j,i]=sim_sig[j,i]*sim_inno[j,i]
        return sim_inno,sim_ret,sim_sig
