# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 10:56:23 2017

@author: chenshengkang
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from pylab import *
fig = plt.figure()
ax1 = fig.add_subplots(2,2,1)
ax2 = fig.add_subplots(2,2,2)
ax3 = fig.add_subplots(2,2,3)
from numpy.random import randn
plt.plot(randn(50).cumsum(), 'k--')
_ = ax1.hist(randn(100),bins = 20,color = 'k' , alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30)+3 * randn(30))

import pandas as pd
from pandas import Series,DataFrame
s = Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
s.plot()

df = DataFrame(np.random.randn(10,4).cumsum(0),
               columns = ['A','B','C','D'],
               index = np.arange(0,100,10))
df.plot()

fig , axes = plt.subplots(2,1)
data = Series(np.random.rand(16),index= list ('abcdefghijklmnop'))
data.plot(kind = 'bar' ,ax = axes[0],color = 'k' , alpha=0.7)
data.plot(kind= 'barh', ax=axes[1],color='k',alpha=0.7)

df = DataFrame(np.random.rand(6,4),
               index = ['one','two','three','four','five','six'],
                columns = pd.Index(['a','b','c','d'],name='Genus'))
df
df.plot(kind='bar')


