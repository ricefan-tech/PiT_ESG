#!/usr/bin/env python
# coding: utf-8

# # Market generator

# In[2]:


from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from cvaetf2 import CVAE
from sklearn.datasets import make_circles
from sklearn import utils


#%% 
#test of CVAE on circles
import time

data, conditions = make_circles(n_samples=10000, noise=0.05, factor=0.5)

scaler=MinMaxScaler(feature_range=(0.00001, 0.99999))
data=scaler.fit_transform(data)
conditions=scaler.fit_transform(conditions.reshape(-1,1))

data = utils.as_float_array(data)
conditions = utils.as_float_array(conditions)

generator = CVAE(input_dim=data.shape[1], n_latent=4, alpha=0.02)
generator.train(data, data_cond=conditions, batch_size=1000, train_size=0.7, n_epochs=2000)

start_time=time.time()
outer_circle_generated = generator.generate((0,), n_samples=1000).numpy()
inner_circle_generated = generator.generate((1,), n_samples=1000).numpy()

end_time=time.time()
t=end_time-start_time
print("Time "+ str(t))
plt.figure()
plt.scatter(outer_circle_generated[:,0],outer_circle_generated[:,1])
plt.scatter(inner_circle_generated[:,0],inner_circle_generated[:,1])
plt.show()


# In[9]:

#from cvae import CVAE
filepath=r"\\ubsprod.msad.ubs.net\userdata\t656703\home\Documents\R\GARCH\SPXVIX.xlsx"

alldata=pd.read_excel(filepath, header=0, sheet_name='SP 500', usecols=['VIX','SP500.Log.Returns'] , skiprows=[1])
data=alldata.iloc[1:,:]
cond=alldata[['VIX']].iloc[:-1]
scaler=MinMaxScaler()
data=scaler.fit_transform(data)

gen=CVAE(input_dim=data.shape[1],n_latent=4, alpha=0.02)
gen.train(data, data_cond=cond, batch_size=10, train_size=0.7, n_epochs=100000)

#%%
import tensorflow as tf

real_spx=data[2816:, 1]
real_vix=cond.iloc[2816:,0]
last_trainvix=cond.iloc[2815,0]

#iteratively generate new paths
pathlength=1200
numpath=50
sims=np.zeros((numpath, pathlength))
for i in range(numpath):
    for j in range(pathlength):
        #first sample is vix forecast
        samp=gen.generate(last_trainvix, 1)
        
        sims[i, j]=samp[0][1]
        last_trainvix=samp[0][0]
        
for i in range(numpath):
    plt.figure()
    plt.plot(sims[i,:])

#%%
plt.figure()
plt.plot(real_spx)
plt.title("train dataset")
