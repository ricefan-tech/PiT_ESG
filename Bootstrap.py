# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:23:52 2020

@author: t656703
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

####################################################################

class HS():
    #pass only levels of data until simstart
    def __init__(self,data, weeks, path, simlength=5):
        self.data=self.scaledata(data)
        self.weeks=weeks
        self.path=path
        self.simlength=simlength
        
    def scaledata(self, data):
        if data.shape[1]>1:
            data.loc[:,"SP500"]=np.log((data.loc[:,"SP500"] / data.loc[:,"SP500"].shift(1)).astype(np.float64))
        
            #first spx value is NaN
            data.iloc[1:,0]=(data.iloc[1:,0].values.reshape(-1,1)/data.iloc[:-1, 1].values.reshape(-1,1))
          
        return data.iloc[1:,0]
            
    def sample(self, data, samples):
        np.random.seed()
        rand_idx=np.random.randint(0, data.shape[0], size=samples)
        return data.iloc[rand_idx]
        
    def BS(self, periodvix):
        # fourth dimension for robustness check
        sims_hs = np.ndarray(shape=(self.simlength * self.weeks, self.path))
        for k in range(self.path):
            sims_hs[:,k:(k+1)]=np.multiply(self.sample(self.data, self.simlength*self.weeks).values.reshape(-1,1), periodvix)
        return sims_hs
   
    
    
    
    
    
    