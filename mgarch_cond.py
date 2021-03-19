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
import copy
class mgarch:
    
    def __init__(self, weeks, path, runs, dist = 'norm'):
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
        self.weeks=weeks
        self.path=path
        self.runs=runs
        
    def process_data(self, d):
        data=copy.deepcopy(d)
        if data.shape[1] > 1:
            data.loc[:,"SP500"] = np.log(data.loc[:,"SP500"] / data.loc[:,"SP500"].shift(1))
            data.loc[:,"VIX"] = np.log(data.loc[:,"VIX"] / data.loc[:,"VIX"].shift(1))
        else:
            print("Data is not multivariate")
        return data.iloc[1:,:]
    
    def fit(self, returns):

        self.rt = np.asarray(returns, dtype=object)
        #should add part that automatically calculates log returns only for return matrix

        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        self.params = np.zeros((self.N, 1, 5))
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        self.mean = self.rt.mean(axis=0)
        self.rt = self.rt - self.mean
        self.sig = np.zeros((self.N, self.T))
        self.inno = np.zeros((self.N, self.T))

        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):

            self.params[i] = self.garch_fit(self.rt[:,i], self.rt[:,-2], i, self.rt[:,-1])
            if i!=self.N-1:
                _,self.sig[i],self.inno[i] = self.garch_var(params=self.params[i][0], returns=self.rt[:,i], fsi=self.rt[:,-1])
            else:
                _, self.sig[i], self.inno[i] = self.garch_var_fsi(self.params[i][0], self.rt[:, i])
            plt.figure()
            plt.plot(self.sig[i], label= "univariate fitted sigma rf "+ str(i))
            plt.show()
            D_t[:,i] = np.sqrt(self.sig[i])
        self.D_t = D_t
        self.D_t = D_t
        a = np.random.uniform(0, 1, 1)
        b = np.random.uniform(0, 1 - a, 1)
        if self.dist == 'norm':
            res = minimize(self.mgarch_loglike, (a, b), args = D_t, method="Nelder-Mead",
            options = {'maxiter':10000, 'disp':True})
            self.a = res.x[0]
            self.b = res.x[1]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'H': self.H_t, 'R': self.R_t} 
        elif self.dist == 't':
            res = minimize(self.mgarch_logliket, (a, b, 3), args = D_t,method="Nelder-Mead",
            options = {'maxiter':10000, 'disp':True})
            self.a = res.x[0]
            self.b = res.x[1]
            self.dof = res.x[2]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'dof': self.dof, 'H': self.H_t, 'R': self.R_t} 
        
    def garch_fit(self, returns, cond, ind, fsi):
        a1=np.random.uniform(0,1,1)
        o=np.random.uniform(0,1,1)
        a2=np.random.uniform(0,1-a1,1)
        #put uniformly 5 params bec otherwise self.param shape makes difficulties, last two params not used for vix and fsi
        initials = (o, a1, a2, 0.5, 0)
        if ind==0:
            #spx 
            initials = (o, a1, a2, 0.5, np.random.uniform(0,1,1))
            res = minimize( self.garch_loglike, x0=initials, args = (returns, cond, fsi),method="Nelder-Mead", options={'disp': True})
        elif ind==1:
            initials = (o, a1, a2, 0.5, np.random.uniform(0,1,1))
            res = minimize( self.garch_loglike, x0=initials, args = (returns, cond, fsi, True),method="Nelder-Mead", options={'disp': True})
        else:
            #fsi
            res = minimize(self.garch_loglike_fsi, x0=initials, args=(returns, cond), method="Nelder-Mead", options={'disp': True})
        return res.x

    def garch_loglike_fsi(self, params, returns, cond):
        T = len(returns)
        sigma2 = self.garch_var_fsi(params, returns)
        # LogL = np.sum(-np.log(2*np.pi*var_t)) - np.sum( (returns.A1**2)/(2*var_t))
        omega = params[0]
        alpha0 = params[1]
        alpha1 = params[2]
        rho = params[3]

        flag, sigma2, i = self.garch_var_fsi(params, returns)

        if flag == True or rho ** 2 > 1:
            return np.inf

        llh_ret = -T / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2) + returns** 2 / sigma2)
        return -llh_ret

    def garch_loglike(self, params, returns, cond,fsi, vix=False):
        T = len(returns)
        omega=params[0]
        alpha0=params[1]
        alpha1=params[2]
        rho=params[3]
        vx1=params[4]
        #flag, sigma2,i =self.garch_var(params, returns,fsi)
        flag, sigma2,i =self.garch_var(params, returns, fsi)
        if (flag==True) | (rho**2>1):
            return np.inf

        llh_ret=-T/2*np.log(2*np.pi)-0.5*np.sum(np.log(sigma2)+(returns)**2/sigma2)
        
        if vix:
            return -llh_ret
        ###use trading day convention
        tau=22
        TT=252
        omega_bar=omega/(1-alpha1)
        alpha_bar=(1-alpha1**TT)/(TT*(1-alpha1))
        c1=omega_bar*(1-alpha_bar)
        vix_model=np.sqrt((c1+sigma2[1:]*alpha_bar)*tau)*100 

        #ix model only has time index 0,...,n-1; bec it needs vola from next time step, hence need to take out last value from VIX as well
        VIX=cond[:-1]
        u=VIX-vix_model

        #covariance of vector of observations is variance of vec
        sig=np.var(u)
        sig_prime=sig*(1-rho**2)
        llh_vix=-T/2*(np.log(2*np.pi)+np.log(sig_prime))+0.5*(np.log(sig_prime)-np.log(sig))-1/(2*sig)*(u[0]**2+np.sum((u[1:]-rho*u[:-1])**2/(1-rho**2)))
        llh=llh_ret+llh_vix
        return -llh

    def garch_var_fsi(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        sigma2 = np.zeros(T)
        inno = np.zeros(T)
        Flag = False
        if (omega <= 0) | (min(alpha, beta) < 0) | (alpha + beta >= 0.9999999):
            Flag = True
            return Flag, sigma2, inno

        for t in range(T):
            if t == 0:
                sigma2[t] = omega / (1 - alpha - beta)
                inno[t] = returns[t] / np.sqrt(sigma2[t])

                if np.isnan(inno[t]) == True:
                    print("VIX inno 0 nan", t, returns[t], sigma2[t])
            else:
                sigma2[t] = omega + alpha * inno[t - 1] ** 2 + beta * sigma2[t - 1]
                inno[t] = returns[t] / np.sqrt(sigma2[t])

            if (sigma2[t] < 0):
                print("neg sigma ", t)
                Flag = True
                return Flag, sigma2, inno
        return Flag, sigma2, inno

    def garch_var(self, params, returns, fsi):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        vx1 = params[4]
        sigma2 = np.zeros(T)
        inno=np.zeros(T)
        Flag=False

        if (omega<=0)| (min(alpha,beta)<0) | (alpha+beta >=0.9999999) | (vx1<0):
            Flag=True
            return Flag, sigma2, inno
        
        for t in range(T):
            if t==0:
                sigma2[t]=omega/(1-alpha-beta)
                inno[t]=returns[t]/np.sqrt(sigma2[t])
                
                if np.isnan(inno[t])==True:
                    print("SPX inno 0 nan",t,  returns[t], sigma2[t])

            else:
                sigma2[t]=omega+alpha*inno[t-1]**2+beta*sigma2[t-1]+vx1*fsi[t-1]**2
                inno[t]=returns[t]/np.sqrt(sigma2[t])
            if (sigma2[t]<0):
                print("neg sigma SPX ", t)
                print(inno[t-1], omega, alpha, beta, vx1)
                break
                Flag=True
                return Flag, sigma2, inno
        return Flag, sigma2, inno
        
    def mgarch_loglike(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        #change type bec it was created as type object
        Q_bar = np.cov(self.rt.astype(np.float32), rowvar=False)

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

        self.Q = Q_t[-1]
        return -loglike

    
    def mgarch_logliket(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        dof = params[2]
        Q_bar = np.cov(self.rt.astype(np.float32), rowvar=False)

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

        self.Q = Q_t[-1]
        return -loglike

    def sim(self, sim_len):
        Q_bar = np.cov(self.rt.astype(np.float32), rowvar=False)

        Q_t = np.zeros((sim_len, self.N, self.N))
        R_t = np.zeros((sim_len, self.N, self.N))
        H_t = np.zeros((sim_len, self.N, self.N))
        D_t = np.zeros((self.N, self.N))
        sig_t = np.zeros((self.N, 1))

        Q_t[0] = self.Q
        R_t[0] = self.R_t[-1]
        # D_T[-1 is last row in D with has last time step std dev. of all assets, np diag turns the vector into diagmatrix
        D_t = np.diag(self.D_t[-1])
        res = np.zeros((sim_len, self.N))
        # = np.matmul(self.rt[0].T/2, self.rt[0]/2)
        res[0, :] = self.rt[-1, :]
        dtinv = np.linalg.inv(D_t)
        et = dtinv * self.rt[-1].T
        #sig has shaoe (1,3)
        sig = self.sig.T[-1]
        inno = self.inno.T[-1]

        for i in range(1, sim_len):
            Q_t[i] = (1 - self.a - self.b) * Q_bar + self.a * (et * et.T) + self.b * Q_t[i - 1]

            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))
            # create next D_t from garch recursion

            for j in range(0,self.N-1):
                #
                #sig_t[j][0] = self.params[j][0][0] + self.params[j][0][1] * inno[j] ** 2 + self.params[j][0][2] * sig[j]
                #print("no fsi ",sig_t[j][0])
                sig_t[j][0] = self.params[j][0][0] + self.params[j][0][1] * inno[j] ** 2 + self.params[j][0][2] * sig[j]+ self.params[j][0][4]*res[i-1,-1]**2
                #print("with fsi ", sig_t[j][0])
            #FSI variance doesnt include FSI
            sig_t[self.N-1][0]=self.params[j][0][0] + self.params[j][0][1] * inno[j] ** 2 + self.params[j][0][2] * sig[j] 
            sig = sig_t
            D_t = np.diag(np.sqrt(sig.flatten()))

            H_t[i] = np.matmul(D_t, np.matmul(R_t[i], D_t))
            et = np.random.normal(size=(self.N, 1))
            #print(et)
            # pdb.set_trace()
            res[i, :] = np.matmul(np.linalg.cholesky(H_t[i]), et).reshape(-1, self.N)
            #no including fsi here as inside this loop res still describes the second equation in the 3 of garch
            inno = (res[i, :].T.reshape(self.N, 1) / np.sqrt(sig))
        #include fsi index in mean function
        #res[:,0]=res[:,0]+res[:,-1]*self.params[0,:,-1]
        if self.dist == 'norm':
            return {'dist': self.dist, 'simulations': res + self.mean, 'covariance': H_t, 'correlation': R_t}
        elif self.dist == 't':
            return {'dist': self.dist, 'dof': self.dof, 'simulations': res + self.mean, 'covariance': H_t,
                    'correlation': R_t}
            
    #simulation producing function
    def GARCH(self,real, simpath, simruns, start_date, simdays=None):
        if simdays==None:
            simdays=self.weeks*5
        tmp=np.zeros(shape=(simdays,3,simpath))  
        
        #start_date gives number of starting week 
        start_spx=real.iloc[start_date*5,0]
        start_vix=real.iloc[start_date*5,1]
        d = self.process_data(real.iloc[:start_date*5,:])
        _ = self.fit(d)
        #print(gar.params)
        tmp[0,0,:]=start_spx
        tmp[0,1,:]=start_vix
        for k in range(simpath):
            final_dict = self.sim(simdays)
            #simulate one year path at once instead of concatenating every week for efficiency purposes
            t= final_dict["simulations"]
            tmp[1:, 0, k]=start_spx*np.exp(t[:-1, 0].cumsum(axis=0).astype(np.float64))
            tmp[1:, 1, k]=start_vix*np.exp(t[:-1, 1].cumsum(axis=0).astype(np.float64))
            tmp[:, 2, k]=t[:, 2]
        return tmp
    
    def cum_ret(self,data):
        cum_ret=np.zeros(shape=data.shape)
        for i in range(data.shape[0]):
            cum_ret[i]=(data[i]-data[0])/data[0]
        return cum_ret

    def strategy_ret(self,final):
        sim_ret=np.zeros(final.shape)
        for i in range(final.shape[-1]):
            for j in range(final.shape[-2]):
                sim_ret[:,:,j,i]=self.cum_ret(final[:,:,j,i])
        return sim_ret
    
    
        
        
        
        
        
        
        