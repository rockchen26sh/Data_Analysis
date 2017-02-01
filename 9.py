# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:21:19 2017

@author: chenshengkang
"""
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
df = DataFrame({'key1':['a','a','b','b','a'],
                'key2':['one','two','one','two','one'],
                'data1':np.random.randn(5),
                'data2':np.random.randn(5)})
df

grouped = df['data1'].groupby(df['key1'])
grouped.mean()

means = df['data1'].groupby([df['key1'],df['key2']]).mean()
means
means.unstack()

states = np.array(['Ohio','California','California','Ohio','Ohio'])
years = np.array([2005,2005,2006,2005,2006]) 
df['data1'].groupby([states,years]).mean()

df.groupby('key1').mean()
df.groupby(['key1','key2']).mean()
df.groupby(['key1','key2']).size()

df.groupby('key1').mean()

for name, group in df.groupby('key1'):
    print (name)
    print (group)
    
for (k1,k2),group in df.groupby(['key1','key2']):
    print(k1,k2)
    print(group)

pieces = dict(list(df.groupby('key1')))
pieces['b']

df.groupby(['key1','key2'])[['data2']].mean()

s_grouped = df.groupby(['key1','key2'])['data2']
s_grouped.mean()

people = DataFrame(np.random.randn(5,5),
                   columns = ['a','b','c','d','e'],
                   index = ['Joe','Steve','Wes','Jim','Travis'])
people.ix[2:3,['b','c']] = np.nan

people

mapping = {'a':'red','b':'red','c':'blue',
           'd':'blue','e':'red','f':'orange'}
by_columns = people.groupby(mapping,axis=1)
by_columns.sum()

map_series = Series(mapping)
map_series
people.groupby(map_series,axis=1).count()

people
people.groupby(len).sum()

key_list = ['one','one','one','two','two']
people.groupby([len,key_list]).min()

columns = pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],
                                     [1,3,5,1,3]],names = ['cty','tenor'])
hier_df = DataFrame(np.random.randn(4,5), columns =columns)
hier_df
hier_df.groupby(level='cty',axis=1).count()


df
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)

def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)
grouped.describe()


tips = pd.read_csv('ch08/tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:5]
tips_grouped = tips.groupby(['sex','smoker'])
grouped_pct = tips_grouped['tip_pct']
grouped_pct.mean()
grouped_pct.agg(['mean','std',peak_to_peak])

grouped_pct.agg([('foo','mean'),('bar',np.std)])

functions = ['count','mean','max']
result = tips_grouped['tip_pct','total_bill'].agg(functions)
result['tip_pct']
ftuples = [('Durchschnitt','mean'),('Abweichung',np.var)]
tips_grouped['tip_pct','total_bill'].agg(ftuples)
tips_grouped.agg({'tip':np.max,'size':'sum'})
tips_grouped.agg({'tip_pct':['count','mean','max'],'size':'sum'})

tips.groupby(['sex','smoker'], as_index=False).mean()

#分组级运算
df
k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means
pd.merge(df,k1_means,left_on='key1',right_index=True)

#transform方法
key = ['one','two','one','two','one']
people.groupby(key).mean()
people.groupby(key).transform(np.mean)

def demean(arr):
    return arr-arr.mean()
demeaned = people.groupby(key).transform(demean)
demeaned.groupby(key).mean()

#apply方法
def top(df,n=5 , column='tip_pct'):
    return df.sort_index(by = column)[-n:]
top(tips,n=6)
tips.groupby('smoker').apply(top)

tips.groupby(['smoker','day']).apply(top,n=1,column='total_bill')
result = tips.groupby('smoker')['tip_pct'].describe()
result
result.unstack('smoker')

tips.groupby('smoker',group_keys=False).apply(top)

#分位数和桶分析
frame = DataFrame({'data1':np.random.randn(1000),
                   'data2':np.random.randn(1000)})
factor = pd.cut(frame.data1,4)
factor

def get_stats(group):
    return {'min':group.min(),'max':group.max(),'count':group.count(),'mean':group.mean()}
grouped = frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()
grouping = pd.qcut(frame.data1,10,labels=False)
grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()

s = Series(np.random.randn(6))
s[::2] = np.nan
s
s.fillna(s.mean())
states = ['Ohio','New York','Vermont','Florida','Oregon','Nevada','California','Idaho']
group_key = ['East'] *4 + ['West']*4
data = Series(np.random.randn(8),index=states)
data
data.groupby(group_key).mean()

fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)
fill_values = {'East':0.5,'West':-1}
fill_func = lambda g:g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)

df = DataFrame({'category':['a','a','a','a','b','b','b','b'],
                'data':np.random.randn(8),
                'weights':np.random.rand(8)})
df
grouped = df.groupby('category')
get_wavg = lambda g:np.average(g['data'],weights=g['weights'])
grouped.apply(get_wavg)

#yahoo finance数据集
close_px = pd.read_csv('ch09/stock_px.csv',parse_dates=True,index_col=0)
close_px[:5]

rets = close_px.pct_change().dropna()
rets
spx_corr = lambda x :x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x:x.year)
by_year.apply(spx_corr)
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))

import statsmodels.api as sm
def regress(data,yvar,xvar):
    Y = data[yvar]
    X = data[xvar]
    X['intercept'] = 1
    result = sm.OLS(Y,X).fit()
    return result.params

by_year.apply(regress,'AAPL',['SPX'])
#pivot_table
tips
tips.pivot_table(index = ['sex','smoker'])
tips.pivot_table(values = ['tip_pct','size'],index = ['sex','day'],columns=['smoker'])
tips.pivot_table(values = ['tip_pct','size'],index = ['sex','day'],columns=['smoker'],margins=True)

tips.pivot_table(values = ['tip_pct'],index=['sex','smoker'],
                 columns = 'day',aggfunc=len,margins=True)

tips.pivot_table(values = ['size'],index=['sex','smoker'],
                 columns = 'day',aggfunc=sum,margins=True,fill_value = 0)

#crosstab
data

#联邦选举委员会
fec = pd.read_csv('ch09/p00000001-ALL.csv')
fec[:5]
fec.ix[123456]

unique_cands =fec.cand_nm.unique()
unique_cands
parties = {'Bachmann, Michelle':'Republican', 
           'Romney, Mitt':'Republican',
           'Obama, Barack':'Democrat',
           "Roemer, Charles E. 'Buddy' III":'Republican',
           'Pawlenty, Timothy':'Republican',
           'Johnson, Gary Earl':'Republican',
           'Paul, Ron':'Republican',
           'Santorum, Rick':'Republican',
           'Cain, Herman':'Republican',
           'Gingrich, Newt':'Republican',
           'McCotter, Thaddeus G':'Republican',
           'Huntsman, Jon':'Republican',
           'Perry, Rick':'Republican'}
fec['party'] = fec.cand_nm.map(parties)
fec[:5]
fec['party'].value_counts()
(fec.contb_receipt_amt > 0) .value_counts()
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo =fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]

fec.contbr_occupation.value_counts()[:10]

occ_mapping = {'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
               'INFORMATION REQUESTED':'NOT PROVIDED',
               'INFORMATION REQUESTED (BEST EFFORTS)':'NOT PROVIDED',
                'C.E.O.':'CEO'}
f = lambda x: occ_mapping.get(x,x)
fec.contbr_occupation = fec.contbr_occupation.map(f)
emp_mapping = {'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
               'INFORMATION REQUESTED':'NOT PROVIDED',
               'SELF' : 'SELF-EMPLOYED',
               'SELF EMPLOYED' : 'SELF-EMPLOYED'}
e = lambda x: emp_mapping.get(x,x)
fec.contbr_employer =fec.contbr_employer.map(e)

by_occupation = fec.pivot_table(values = ['contb_receipt_amt'],
                                index = ['contbr_occupation'],
                                columns = 'party',aggfunc=sum)
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm
over_2mm.plot(kind='barh',figsize = (12,8))

def get_top_am(group,key,n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.order(ascending = False)[:n]

grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_am,'contbr_occupation',n=7)
grouped.apply(get_top_am,'contbr_employer',n=10)

bins = np.array([0,1,10,100,1000,10000,100000,1000000,10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt,bins)
labels
grouped = fec_mrbo.groupby(['cand_nm',labels])
grouped.size().unstack(0)

bucket_sums =grouped.contb_receipt_amt.sum().unstack(0)
bucket_sums

normed_sums = bucket_sums.div(bucket_sums.sum(axis=1),axis=0)
normed_sums
normed_sums[:-2].plot(kind='barh',stacked=True)
normed_sums.plot(kind='barh')

grouped = fec_mrbo.groupby(['cand_nm','contbr_st'])
totals =grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals[:10]

percent = totals.div(totals.sum(1),axis=0)
percent[:10]

from mpl_toolkits.basemap import Basemap,cm
