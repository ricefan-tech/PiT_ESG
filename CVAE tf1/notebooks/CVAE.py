#!/usr/bin/env python
# coding: utf-8

# # CVAE

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler

import base
import cvae
import importlib
importlib.reload(cvae)


# ## Create the data

# In[8]:


data, conditions = make_circles(n_samples=10000, noise=0.05, factor=0.5)

# We normalise the data to [0, 1]^2
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


# In[9]:


# Plot dataset
plt.scatter(*data.T, c=conditions, s=2)
plt.show()


# # Train the CVAE

# In[10]:


generator = cvae.CVAE(n_latent=4, alpha=0.02)
generator.train(data, data_cond=conditions.reshape(-1, 1),
                n_epochs=10000)


# # Generate samples

# In[13]:


outer_circle_generated = generator.generate((0,), n_samples=1000)
inner_circle_generated = generator.generate((1,), n_samples=1000)


# In[14]:


plt.scatter(*outer_circle_generated.T, s=2)
plt.scatter(*inner_circle_generated.T, s=2)
plt.show()


# In[ ]:




