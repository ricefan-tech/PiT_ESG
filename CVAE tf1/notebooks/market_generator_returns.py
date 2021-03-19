#!/usr/bin/env python
# coding: utf-8

# # Market generator

# In[2]:


from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.datasets import make_circles

from sklearn.preprocessing import MinMaxScaler
from cvae_tf1 import CVAE

data, conditions = make_circles(n_samples=10000, noise=0.05, factor=0.5)

scaler=MinMaxScaler(feature_range=(0.00001, 0.99999))
data=scaler.fit_transform(data)
conditions=scaler.fit_transform(conditions.reshape(-1,1))

#data = utils.as_float_array(data)
#conditions = utils.as_float_array(conditions)

generator = CVAE( n_latent=4, alpha=0.02)
generator.train(data, data_cond=conditions, n_epochs=2000, show_progress=True)

outer_circle_generated = generator.generate((0,), n_samples=1000)
inner_circle_generated = generator.generate((1,), n_samples=1000)
plt.figure()
plt.scatter(outer_circle_generated[:,0],outer_circle_generated[:,1])
plt.scatter(inner_circle_generated[:,0],inner_circle_generated[:,1])
plt.show()

# In[9]:


import importlib
importlib.reload(market_generator)


# In[10]:


filepath=r"\\ubsprod.msad.ubs.net\userdata\t656703\home\Documents\R\GARCH\SPXVIX.xlsx"
MG = market_generator.MarketGenerator(filepath,freq="W", sig_order=None)


# ## Plot paths

# In[11]:

for i in range(MG.rf):
    for path in MG.windows:
        #::2 gives every second element, starting with 0,2,4 etc
        #columns inside MG windows are lag1, lag1; lag1, lead1; lag2=lead1, lag2; lag2, lead2 etc, hence take every second row of lead columns. (lag columns are again divided in rf1, rf2 etc dep. on code)
        #so first MG.rf columns are the lags
        #paths are scaled by first value of oberservation period to standardized around 1, so that more extreme values are better visible
    
        returns = path[::2, MG.rf + i]/ path[0, MG.rf + i]
        plt.plot(returns, "b", alpha=0.05)
    
    plt.title("{} frequency paths real data of risk factor {}".format(MG.freq, i+1))
    plt.xlabel("Days")
    plt.show()




# In[12]:
# ## Train generative model

MG.train(n_epochs=100000)


# ## Generate

# In[13]:


generated = np.array([MG.generate(cond) for cond in MG.conditions])
#generated = MG.generate(MG.conditions[100], n_samples=len(MG.logsigs))


# In[14]:

len_freq=int(generated.shape[1]/2)
PROJECTIONS = [(0, 2), (2, 3), (1, 0), (0, 3)]
#PROJECTIONS = [(0, 2), (2, 3), (1, 0), (0, 3), (1,2),(1,3)]


#if only returns are used, then there is no influence of lead lag transformation left in MG.orig_logsig
for j in range(MG.rf):
    index=1
    #f,ax=plt.subplot(4,1, index)
    plt.figure(figsize=(12, 8))        
    for i, projection in enumerate(PROJECTIONS):
            
        plt.subplot(2,2,i+1)
        col1=j*len_freq+projection[0]
        col2=j*len_freq+projection[1]
        plt.scatter(MG.orig_logsig[:, col1], MG.orig_logsig[:, col2],
                    label="Real data")
        plt.scatter(generated[:, col1], generated[:, col2],
                   label="Generated data")
        plt.xlabel(projection[0], fontsize=14)
        plt.ylabel(projection[1], fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.title('Risk Factor {} Freq {}'.format(j+1, MG.freq))
        plt.legend(('Real Data', 'Generated Data'))
        index+=1
        #plt.legend()
    
    plt.show()

#%%

from statsmodels.graphics.gofplots import qqplot_2samples
import statsmodels.api as sm

pp_x = sm.ProbPlot(MG.orig_logsig[:, 1].T)
pp_y = sm.ProbPlot(generated[:, 1].T)
fig=qqplot_2samples(pp_x,pp_y, xlabel="Quantiles of Data", ylabel="Quantiles of CVAE", line="45")
fig.suptitle("CVAE vs. Data sec day of week distribution Risk Factor 1")
# ## Validation: two-sample statistical test



# In[15]:

for i in range(MG.rf):
    col2=int((i+1)*len_freq)
    col1=int(i*len_freq)
    M1=MG.orig_logsig[:,col1:col2].cumsum(axis=1).astype(np.float64)
    M2=generated[:,col1:col2].cumsum(axis=1).astype(np.float64)
    paths_orig=np.exp(M1)
    paths_generated = np.exp(M2)
    plt.figure()
    for p1, p2 in zip(paths_orig, paths_generated):
        #fancy wy of plotting every row
        plt.plot(np.r_[1., p1], "C0", alpha=0.2)
        plt.plot(np.r_[1., p2], "C1", alpha=0.2)
        plt.legend(("Original Paths", "Generated Paths"))
    plt.title("Paths of risk factor {} frequency {}".format(i+1, MG.freq))
    plt.show()


# In[16]:


from utils.leadlag import leadlag
import process_discriminator
import iisignature

order = 4
sigs1 = np.array([np.r_[1., iisignature.sig(leadlag(p), order)] for p in tqdm(paths_generated[:-1])])
sigs2 = np.array([np.r_[1., iisignature.sig(leadlag(p), order)] for p in tqdm(paths_orig)])


# In[17]:


import importlib
importlib.reload(process_discriminator)


# In[18]:


res = process_discriminator.test(sigs1, sigs2, order=order, compute_sigs=False,
                                 confidence_level=0.9999)

print("Are the generated and real distributions DIFFERENT? {}".format(res))


# In[ ]:




