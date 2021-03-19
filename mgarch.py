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

class mgarch:
    
    def __init__(self, dist = 'norm'):
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
            
    def garch_fit(self, returns):
        a1=np.random.uniform(0,1,1)
        o=np.random.uniform(0,1,1)
        a2=np.random.uniform(0,1-a1,1)
        initials=(o,a1,a2)
        res = minimize( self.garch_loglike, x0=initials, args = returns,method="Nelder-Mead", options={'disp': True})
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        sigma2 = self.garch_var(params, returns)
        flag, sigma2,i =self.garch_var(params, returns)

        if flag:
            return np.inf
        
        llh_ret=-T/2*np.log(2*np.pi)-0.5*np.sum(np.log(sigma2)+returns**2/sigma2)
    
        return -llh_ret
        
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
                sigma2[t]=omega/(1-alpha-beta)
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
      
        return Flag, sigma2, inno
        
    def mgarch_loglike(self, params, D_t):
 
        a = params[0]
        b = params[1]
        Q_bar = np.cov(self.rt.T)

        Q_t = np.zeros((self.T,self.N,self.N))
        self.R_t = np.zeros((self.T,self.N,self.N))
        self.H_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

        loglike = 0
                
        for i in range(1,self.T):
            if (min(a,b)<0) | (a+b>=1):
                return np.inf

            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))
           
            self.R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


            self.H_t[i] = np.matmul(dts, np.matmul(self.R_t[i], dts))   

            loglike = loglike + self.N*np.log(2*np.pi) + \
                      2*np.log(D_t[i].sum()) + \
                      np.log(np.linalg.det(self.R_t[i])) + \
                      np.matmul(self.rt[i], (np.matmul( np.linalg.inv(self.H_t[i]), self.rt[i].T)))
        self.Q=Q_t[-1]

        return -loglike

    
    def mgarch_logliket(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        dof = params[2]
        Q_bar = np.cov(self.rt.T)

        Q_t = np.zeros((self.T,self.N,self.N))
        self.R_t = np.zeros((self.T,self.N,self.N))
        self.H_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

        loglike = 0
        for i in range(1,self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            self.R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


            self.H_t[i] = np.matmul(dts, np.matmul(self.R_t[i], dts))   

            loglike = loglike + np.log( gamma((self.N+dof)/2.)) - np.log(gamma(dof/2)) \
                      -(self.N/2.)*np.log(np.pi*(dof - 2)) - np.log(np.linalg.det(self.H_t[i])) \
- ((dof+ self.N)*( ((np.matmul(self.rt[i], (np.matmul( np.linalg.inv(self.H_t[i]), self.rt[i].T))))/(dof - 2.)) + 1)/2.)


        return -loglike
    
    
    def predict(self, ndays = 1):
        #one step forecast for every time step in data
        if 'a' in dir(self):
            #Q_pd=pd.DataFrame(data=self.rt)
            #Q_bar=Q_pd.cov().to_numpy()
            Q_bar = np.cov(self.rt.reshape(self.N, self.T))
            Q_t = np.zeros((self.T,self.N,self.N))
            R_t = np.zeros((self.T,self.N,self.N))
            H_t = np.zeros((self.T,self.N,self.N))

            Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

            
            for i in range(1,self.T):
                dts = np.diag(self.D_t[i])
                dtinv = np.linalg.inv(dts)
                et = dtinv*self.rt[i].T
                Q_t[i] = (1-self.a-self.b)*Q_bar + self.a*(et*et.T) + self.b*Q_t[i-1]
                qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

                R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


                H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))  

            if self.dist == 'norm':
                return {'dist': self.dist, 'cov': H_t[-1]*np.sqrt(ndays)}
            elif self.dist == 't':
                return {'dist': self.dist, 'dof': self.dof, 'cov': H_t[-1]*np.sqrt(ndays)}
            
        else:
            print('Model not fit')
            
    def sim(self, sim_len):
        Q_bar = np.cov(self.rt.T)
        
        Q_t = np.zeros((sim_len,self.N,self.N))
        R_t = np.zeros((sim_len,self.N,self.N))
        H_t = np.zeros((sim_len,self.N,self.N))
        D_t = np.zeros((self.N,self.N))
        sig_t=np.zeros((self.N,1))
        
        Q_t[0]=self.Q
        R_t[0]=self.R_t[-1]
        #D_T[-1 is last row in D with has last time step std dev. of all assets, np diag turns the vector into diagmatrix
        D_t=np.diag(self.D_t[-1])
        res=np.zeros((sim_len, self.N))
        # = np.matmul(self.rt[0].T/2, self.rt[0]/2)
        res[0,:]=self.rt[-1,:]
        dtinv=np.linalg.inv(D_t)
        et = dtinv*self.rt[-1].T
        
        sig=self.sig.T[-1]
        inno=self.inno.T[-1]
        
        for i in range(1,sim_len):
            Q_t[i]=(1-self.a-self.b)*Q_bar + self.a*(et*et.T) + self.b*Q_t[i-1]
           
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))
            
            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))
            #create next D_t from garch recursion
           
            for j in range(self.N):
                sig_t[j][0]=self.params[j][0][0]+self.params[j][0][1]*inno[j]**2+self.params[j][0][2]*sig[j]
            sig=sig_t
            D_t=np.diag(np.sqrt(sig.flatten()))
            
            H_t[i] = np.matmul(D_t, np.matmul(R_t[i], D_t))  
            et=np.random.normal(size=(self.N, 1))
            #pdb.set_trace()
            res[i,:]=np.matmul(np.linalg.cholesky(H_t[i]),et).reshape(-1,self.N)
            
            inno=(res[i,:].T.reshape(self.N,1)/ np.sqrt(sig))
           
       

        if self.dist == 'norm':
            return {'dist': self.dist, 'simulations': res+self.mean, 'covariance': H_t, 'correlation': R_t}
        elif self.dist == 't':
            return {'dist': self.dist, 'dof': self.dof, 'simulations': res+self.mean, 'covariance': H_t, 'correlation': R_t}
            
    def fit(self, returns):
        
        self.rt = np.asarray(returns, dtype=np.float32)
        #should add part that automatically calculates log returns only for return matrix
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        
        self.params=np.zeros((self.N, 1,3))
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        self.sig=np.zeros((self.N, self.T))
        self.inno=np.zeros((self.N, self.T))
        
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            self.params[i]= self.garch_fit(self.rt[:,i])
            _,self.sig[i],self.inno[i] = self.garch_var(self.params[i][0], self.rt[:,i])
            D_t[:,i] = np.sqrt(self.sig[i])
        self.D_t = D_t
        a=np.random.uniform(0,1,1)
        b=np.random.uniform(0,1-a,1)
        if self.dist == 'norm':
            
            res = minimize(self.mgarch_loglike, (a,b), args = D_t, method="Nelder-Mead",
            options = {'maxiter':100000, 'disp':True}
            )
            self.a = res.x[0]
            self.b = res.x[1]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'H': self.H_t, 'R': self.R_t} 
        elif self.dist == 't':
            res = minimize(self.mgarch_logliket, (a,b, 3), args = D_t,
            method="Nelder-Mead", options = {'maxiter':10000000, 'disp':True}
            )
            self.a = res.x[0]
            self.b = res.x[1]
            self.dof = res.x[2]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'dof': self.dof, 'H': self.H_t, 'R': self.R_t} 