# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 09:39:24 2017

@author: chenshengkang
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
#Series数组
obj = Series([4,7,-5,3])
obj
obj.values
obj.index
#制定index
obj2 = Series([4,7,-5,3], index = ['d','b','a','c'])
obj2
obj2['a']
obj2['c'] = 6
obj2
obj2[['a','c','d']]

obj2[obj2>0]
obj2*2
np.exp(obj2)

'b' in obj2
'e' in obj2

sdata = {'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3 = Series(sdata)
obj3
states= {'California','Ohio','Oregon','Texas'}
obj4 = Series(sdata,index=states)
obj4

pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

#算术运算中自动对齐index
obj3 + obj4

#列名赋值
obj4.name = 'population'
obj4.index.name = 'state'
obj4

#dataframe
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],'year':[2000,2001,2002,2001,2002],'pop':[1.5,1.7,3.6,2.4,2.9]}
frame = DataFrame(data)
frame

DataFrame(data,columns = ['year','state','pop'])
frame2 = DataFrame(data,columns = ['year','state','pop','debt'],
                   index = ['one','two','three','four','five'])
frame2
frame2.columns
frame2['state']
frame2.ix['three']

frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(5.)

val = Series([-1.2,-1.5,-1.7],index = ['two','four','five'])
frame2['debt'] = val
frame2
frame2['eastern'] = frame2.state == 'Ohio'
frame2
del frame2['eastern']
frame2

frame2.columns

pop = {'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3 = DataFrame(pop)
frame3
frame3.T
DataFrame(pop,index=[2001,2002,2003])
pdata = {'Ohio':frame3['Ohio'][:-1],'Nevada':frame3['Nevada'][:2]}
DataFrame(pdata)

frame3.index.name = 'year'
frame3.columns.name='state'
frame3
frame3.values

obj = Series(range(3),index=['a','b','c','d'])
index = obj.index
index
index[1:]
index = pd.Index(np.arange(3))
obj2 = Series([1.5,-2.5,0],index=index)
obj2.index is index

frame3
'Ohio' in frame3.columns
2003 in frame3.index

obj= Series([4.5,7.2,-5.3,3.6], index=['d','b','a','c'])
obj
obj2 = obj.reindex(['a','b','c','d','e'])
obj2
obj.reindex(['a','b','c','d','e'],fill_value = 0)
obj3 = Series(['blue','purple','yellow'],index=[0,2,4])
obj3.reindex(range(6),method='ffill')

frame = DataFrame(np.arange(9).reshape(3,3),index=['a','c','d'],columns=['Ohio','Texas','California'])
frame
frame2 = frame.reindex(['a','b','c','d'])
frame2

states = ['Texas','Utah','California']
frame.reindex(columns = states)
frame.reindex(index=['a','b','c','d'],method='ffill',columns=states)
frame.ix[['a','b','c','d'],states]

obj = Series(np.arange(5.),index=['a','b','c','d','e'])
new_obj = obj.drop('c')
new_obj
obj.drop(['d','c'])

data = DataFrame(np.arange(16).reshape((4,4)),
                 index=['Ohio','Colorado','Utah','New York'],
                 columns = ['one','two','three','four'])
data.drop(['Colorado','Ohio'])
data

obj = Series(np.arange(4.),index = ['a','b','c','d'])
obj['b']
obj[2:4]
obj[obj<2]

obj['b':'c'] = 5
obj

data = DataFrame(np.arange(16).reshape((4,4)),
                 index=['Ohio','Colorado','Utah','New York'],
                 columns = ['one','two','three','four'])
data['two']
data[['two','one']]
data[:2]
data[data['three'] > 5]
data<5
data[data<5] =0
data
data.ix['Colorado',['two','three']]
data.ix[['Colorado','Utah'],[3,0,1]]
data.ix[data.three>5, :3]

s1 = Series([7.3,-2.5,3.4,1.5],index = ['a','c','d','e'])
s2 = Series([-2.1,3.6,-1.5,4,3.1],index = ['a','c','e','f','g'])
s1+s2

df1 = DataFrame(np.arange(12.).reshape((3,4)),columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4,5)),columns=list('abcde'))
df1
df2
df1+df2
df1.add(df2.fill_value=0)
df1

frame = DataFrame(np.random.randn(4,3),columns = list('bde'),index=['Utah','Ohio','Texas','Oregon'])
np.abs(frame)
f = lambda x : x.max() - x.min()
frame.apply(f)
frame.apply(f,axis=1)

def f(x):
    return Series([x.min(),x.max()],index=['min','max'])
frame.apply(f)

format=lambda x: '%.2f' % x
frame.applymap(format)
frame['e'].map(format)

obj = Series(range(4), index=['d','a','b','c'])
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2,4)),index=['three','one'],columns=['d','a','b','c'])
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1,ascending=False)
obj=Series([4,7,-3,2])
obj.order()
frame.sort_index(by='b')
frame.sort_index(by = ['a','b'])

obj = Series([7,-5,7,4,2,0,4])
obj.rank()
obj.rank(method = 'first')
obj.rank(ascending=False,method='max')

#汇总和计算描述统计
df = DataFrame([[1.3,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]])
df.index = ['a','b','c','d']
df.columns=['one','two']
df.sum()
df.sum(axis = 1)
df.mean(axis=1,skipna=False)
df.idxmax()
df.cumsum()
df.describe()

#相关系数 协方差
import pandas.io.data as web
all_data = {}
for ticker in ['AAPL','IBM','MSFT','GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker,'1/1/2000','1/1/2010')

price = DataFrame({tic:data['Adj Close']
                   for tic, data in all_data.items()})

volume = DataFrame({tic: data['Volume']
                    for tic, data in all_data.items()})
returns = price.pct_change() #百分数变换
returns.MSFT.corr(returns.IBM) #计算两组数据的相关系数
returns.MSFT.cov(returns.IBM) #计算两组数据的协方差

returns.corr() #计算各列的相关系数
returns.cov() #计算各列的协方差

returns.corrwith(returns.IBM) #计算各列与另外一个知道的series或dataframe之间的相关系数
returns.corrwith(volume)

#唯一值、值计数以及成员资格
obj = Series(['c','a','d','a','a','b','b','c','c'])
obj.unique()
obj.value_counts()
pd.value_counts(obj.values,sort=False)
uniques = obj.unique()
uniques[1] = 'f'
mask = obj.isin(uniques)

data = DataFrame({'Qu1':[1,2,2,3,4],
                  'Qu2':[2,3,1,3,4],
                  'Qu3':[1,5,2,4,4]})
result = data.apply(pd.value_counts).fillna(0)
result

#缺失值处理
string_data = Series(['aardvark','artichoke',np.nan,'avocado'])
string_data

string_data[0] = None
string_data.isnull()
string_data.notnull()

data = Series([1,np.nan,3.5,np.nan,7])
data.dropna()
data = DataFrame([[1.,6.5,3.],[1,np.nan,np.nan],
                 [np.nan,np.nan,np.nan],[np.nan,6.5,3.]])
data
data.dropna(how='all')
data.dropna(how='all',axis=1)

df = DataFrame(np.random.randn(7,3))
df
df.ix[:4,1] = np.nan
df.ix[:2,2] = np.nan
df
df.dropna()
df.dropna(thresh=3)

df.fillna(0)
df.fillna({1:0.5,2:-1})

#层次化索引
data = Series(np.random.randn(10),index=[['a','a','a','b','b','b','c','c','d','d'],
              [1,2,3,1,2,3,1,2,2,3]])
data
data.index
data[['b','d']]
data[:,2]

data.unstack()
data.unstack().stack()

frame = DataFrame(np.arange(12).reshape((4,3)),
                  index = [['a','a','b','b'],[1,2,1,2]],
                   columns = [['Ohio','Ohio','Colorado'],
                              ['Green','Red','Green']])
frame
frame.index.names = ['key1','key2']
frame.columns.names = ['state','color']
frame
MultiIndex.from_arrays([['Ohio','Ohio','Colorado'],['Green','Red','Green']],
                       names = ['state','color'])

frame.swaplevel('key1','key2')
frame.sortlevel(1)
frame.swaplevel(0,1).sortlevel(0)
frame.sum(level='key2')
frame.sum(level = 'color',axis = 1)

frame2 = frame.stack()
frame3 = frame2.stack()
frame3
frame3.unstack('key1')
frame3.reset_index()


