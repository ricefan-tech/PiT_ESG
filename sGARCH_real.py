# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:32:57 2020

@author: t656703
"""

from scipy.optimize import minimize
import numpy as np
from scipy.special import gamma
from matplotlib import pyplot as plt
import pandas as pd
import pdb
from scipy.special import comb, gamma, gammainc, gammaincc, gammaln

class sgarch:
    
    def __init__(self, dist = 'norm'):
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
            
    def garch_fit(self, returns):
        np.random.seed()
        a1=np.random.uniform(0,1,1)
        o=np.random.uniform(0,1,1)
        a2=np.random.uniform(0,1-a1,1)
        initials=(o,a1,a2)
        self.data_mean=np.mean(returns[:,0])
        # print("initials: ", initials)
        #initials=(0.15, 0.6, 0.05)
        if self.dist=="norm":
            res = minimize( self.garch_loglike, x0=initials, args = returns,method="Nelder-Mead", options={'maxiter':5000,'disp': True})
        else:
            res = minimize( self.garch_logliket, x0=initials, args = returns,method="Nelder-Mead", options={'maxiter':5000,'disp': True})
        self.omega=res.x[0]
        self.alpha=res.x[1]
        self.beta=res.x[2]
        
        return res.x

    def garch_loglike(self, params, inpt):
        returns=inpt[:,0]-self.data_mean
        #returns=returns-self.data_mean
        #cond=inpt[:,1]
        T = len(returns)
        flag, sigma2,i =self.garch_var(params, returns)
        if flag:
            return np.inf
        llh_ret=-T/2*np.log(2*np.pi)-0.5*np.sum(np.log(sigma2)+returns**2/sigma2)
        llh_vix=self.vix_loglike(params, inpt, sigma2)
        return -(llh_ret+llh_vix)
    
    
    def garch_logliket(self, params, inpt, nu=4):
        returns=inpt[:,0]-self.data_mean
        #returns=returns-self.data_mean
        #cond=inpt[:,1]
        flag, sigma2,i =self.garch_var(params, returns)        
        if flag:
            return np.inf 
        lls = gammaln((nu + 1) / 2) - gammaln(nu / 2) - np.log(np.pi * (nu - 2)) / 2
        lls -= 0.5 * (np.log(sigma2))
        lls -= ((nu + 1) / 2) * (np.log((1 + (returns ** 2.0) / (sigma2 * (nu - 2))).astype(np.float64)))
        llh_ret=np.sum(lls)
        llh_vix=self.vix_loglike(params, inpt, sigma2)
        
        return -(llh_ret+llh_vix)
    
    def vix_loglike(self, params, inpt, sigma2, rho=0.5):
        omega=params[0]
        alpha=params[1]
        beta=params[2]
        ret=inpt[:,0]
        cond=inpt[:,1]
        n=inpt.shape[0]
        tau=22
        T=252
        self.sigma_bar=omega/(1-beta)
        self.gamma=(1-beta)**T/(T*(1-beta))
        vix_model=(self.sigma_bar*(1-self.gamma)+sigma2[1:]*self.gamma)
        vix_model=np.sqrt(vix_model*tau)*100        
        #ix model only has time index 0,...,n-1; bec it needs vola from next time step, hence need to take out last value from VIX as well
        VIX=cond[:-1]
        u=VIX-vix_model
        
        ########checking equation 10 validity######################
        # spot=(1/self.gamma)*(1/tau*(cond[:-1]/100)**2-self.sigma_bar*(1-self.gamma))
        # if np.any(spot<0):
        # #if spot[-1]<0:
        #     #print("negative spot")
        #     return np.inf
        #covariance of vector of observations is variance of vec
        sig=np.var(u)
        sig_prime=sig*(1-rho**2)
        llh_vix=-n/2*(np.log(2*np.pi)+np.log(sig_prime))+0.5*(np.log(sig_prime)-np.log(sig))-1/(2*sig)*(u[0]**2+np.sum((u[1:]-rho*u[:-1])**2/(1-rho**2)))  
        return llh_vix


    
    def garch_var(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        sigma2 = np.zeros(T)     
        inno=np.zeros(T)
        Flag=False
     
        ##for return part#####################################
        if (omega<=0)| (min(alpha,beta)<0) | (alpha+beta >=1.0):
            Flag=True
            return Flag, sigma2, inno
        
        for t in range(T):
            if t==0:
                sigma2[t]=np.var(returns)
                inno[t]=returns[t]/np.sqrt(sigma2[t])
                
                if np.isnan(inno[t])==True:
                    print("inno 0 nan",t,  returns[t], sigma2[t])
              
            else:
                sigma2[t]=omega+alpha*inno[t-1]**2+beta*sigma2[t-1]
                inno[t]=returns[t]/np.sqrt(sigma2[t])
    
            if (sigma2[t]<0):
                print("neg sigma ", t)
                Flag=True
                return Flag, sigma2, inno
        self.inno=inno
        self.sigma2=sigma2
        return Flag, sigma2, inno
    
    
 
  
    def sim(self, params, num_path, path_length,  nu=4):
        np.random.seed(1)
        
        omega=params[0]
        alpha=params[1]
        beta=params[2]
        if self.dist=="norm":
            sim_inno=np.random.normal(size=(path_length,num_path))
            print(np.mean(sim_inno))
        else:
            sim_inno=np.random.standard_t(df=nu, size=(path_length,num_path))
        sim_ret=np.zeros(shape=(path_length,num_path))
        sim_sig=np.zeros(shape=(path_length,num_path))
        
        for i in range(num_path):
            for j in range(1,path_length):
                sim_sig[j,i]=omega+alpha*sim_inno[(j-1),i]**2+beta*sim_sig[(j-1),i]
                sim_ret[j,i]=np.sqrt(sim_sig[j,i])*sim_inno[j,i]
            
        return sim_inno, sim_ret, sim_sig
    
    
    def generate(self, startvol,starts, num_path, path_length,  nu=4):
        np.random.seed()
        
        if self.dist=="norm":
            sim_inno=np.random.normal(size=(path_length,num_path))
        else:
            sim_inno=np.random.standard_t(df=nu, size=(path_length,num_path))
        sim_ret=np.zeros(shape=(path_length,num_path))
        sim_sig=np.zeros(shape=(path_length,num_path))
        #sim_ret[0,:]=startret
        #sim_sig[0,:]=startvol
        for i in range(num_path):
            for j in range(path_length):
                if j==0:
                    sim_sig[j,i]=self.omega+self.alpha*self.inno[starts]**2+self.beta*startvol
                else:
                    sim_sig[j,i]=self.omega+self.alpha*sim_inno[(j-1),i]**2+self.beta*sim_sig[(j-1),i]
                #sim_ret[j,i]=np.sqrt(sim_sig[j,i])*sim_inno[j,i]+self.data_mean
                sim_ret[j,i]=np.sqrt(sim_sig[j,i])*sim_inno[j,i]+self.data_mean
                #print(sim_ret[j,i])
                #print(self.data_mean)
        return sim_inno, sim_ret, sim_sig
  