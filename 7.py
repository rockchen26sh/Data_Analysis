# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:33:04 2017

@author: chenshengkang
"""
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
df1 = DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
df2 = DataFrame({'key':['a','b','d'],'data2':range(3)})
df1
df2

pd.merge(df1,df2)
pd.merge(df1,df2,on='key')

df1 = DataFrame({'lkey':['b','b','a','c','a','a','b'],'data1':range(7)})
df2 = DataFrame({'rkey':['a','b','d'],'data2':range(3)})
pd.merge(df1,df2,left_on='lkey',right_on='rkey')

df1 = DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
df2 = DataFrame({'key':['a','b','d'],'data2':range(3)})
pd.merge(df1,df2,how='outer')

left = DataFrame({'key1':['foo','foo','bar'],
                  'key2':['one','two','one'],
                  'lval':[1,2,3]})
left
right = DataFrame({'key1':['foo','foo','bar','bar'],
                   'key2':['one','one','one','two'],
                   'rval':[4,5,6,7]})
right

pd.merge(left,right,on=['key1','key2'],how='outer')
pd.merge(left,right,on='key1')
pd.merge(left,right,on='key1',suffixes = ('_left','_right'))

s1 = Series([0,1], index = ['a','b'])
s2 = Series([2,3,4], index = ['c','d','e'])
s3 = Series([5,6], index = ['f','g'])
pd.concat([s1,s2,s3])
pd.concat([s1,s2,s3],axis = 1)
s4 = pd.concat([s1*5 , s3])
s4
pd.concat([s1,s4], axis=1)
pd.concat([s1,s4], axis =1,join='inner')
pd.concat([s1,s4], axis = 1 ,join_axes=[['a','c','b','e']])

pd.concat([s1,s1,s3])
result = pd.concat([s1,s1,s3],keys=['one','two','three'])
result

result.unstack()

pd.concat([s1,s1,s4], axis=1)
pd.concat([s1,s1,s3],axis=1,keys=['one','two','three'])

df1 = DataFrame(np.arange(6).reshape(3,2),index=['a','b','c'],columns=['one','two'])
df2 = DataFrame(5 + np.arange(4).reshape(2,2),index = ['a','c'],columns = ['three','four'])
df1
df2

pd.concat([df1,df2],axis=1,keys=['level1','level2'])
pd.concat({'level':df1,'level2':df2},axis=1)
pd.concat([df1,df2],axis=1,keys=['level1','level2'],names=['upper','lower'])

df1 = DataFrame(np.random.randn(3,4),columns = ['a','b','c','d'])
df2 = DataFrame(np.random.randn(2,3),columns = ['b','d','a'])
df1
df2
pd.concat([df1,df2],ignore_index=True)

a = Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],
           index=['f','e','d','c','b','a'])
b = Series(np.arange(len(a),dtype=np.float64),
           index = ['f','e','d','c','b','a'])
b[-1] = np.nan
a
b
np.where(pd.isnull(a),b,a)
a[2:]
b[:-2]
b[:-2].combine_first(a[2:])

df1 = DataFrame({'a':[1.,np.nan,5.,np.nan],
                 'b':[np.nan,2.,np.nan,6.],
                 'c':range(2,18,4)})
df2 = DataFrame({'a':[5.,4.,np.nan,3.,7.],
                 'b':[np.nan,3.,4.,6.,8.]})
df1
df2
df1.combine_first(df2)

data = DataFrame(np.arange(6).reshape((2,3)),
                 index = pd.Index(['Ohio','Colorado'],name = 'state'),
                columns = pd.Index(['one','two','three'],name='number'))
data
result = data.stack()
result
result.unstack()

s1 = Series([0,1,2,3],index = ['a','b','c','d'])
s2 = Series([4,5,6],index = ['c','d','e'])
data2 = pd.concat([s1,s2],keys=['one','two'])
data2
data2.unstack()
data2.unstack().stack()

df = DataFrame({'left':result,'right':result+5},
               columns = pd.Index(['left','right'],name='side'))
df
df.unstack('state')
df.unstack()
df.unstack('state').stack('side')


data = DataFrame({'k1':['one'] *3 + ['two'] *4,'k2':[1,1,2,3,3,4,4]})
data
data.duplicated()
data.drop_duplicates()
data['v1'] = range(7)
data.drop_duplicates(['k1'])
