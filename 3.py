# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:51:14 2017

@author: cck{28488747}
"""
import pandas as pd
import numpy as np
#读取1880出生数据
names_1880 = pd.read_csv("ch02/names/yob1880.txt",
                         header=None,
                         names = ['name','sex','births'])
#按性别分组计算出生人数
names_1880.groupby('sex').births.sum()

#批量读取
years = range(1880,2011)
col = ['name','sex','births']
total_names = []

for year in years:
    path = "ch02/names/yob" + str(year) + '.txt'
    frame = pd.read_csv(path,header=None,names = col)
    frame['year'] = year
    total_names.append(frame)

#合并data_frame
total_names = pd.concat(total_names,ignore_index=True)#按行组合data_frame
#各年出生人数
total_names.pivot_table(values='births',
                        index='year',
                        columns='sex',
                        aggfunc=sum)
total_births= pd.pivot_table(total_names,
               index='year',
               columns='sex',
               values='births',
               aggfunc=sum)
#输出图
total_births.plot(title='Total_births')

#相对比例prop
def add_prop(group):
    births = group.births.astype(float)
    
    group['prop'] = births/births.sum()
    return group
total_names = total_names.groupby(['year','sex']).apply(add_prop)
total_names.groupby(['year','sex']).prop.sum()

#筛选出生人数前1000条记录
top1000 = total_names.sort_index(by='births',ascending=False)[:1000]

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table(index='year',
                                   columns='name',
                                   values='births',
                                   aggfunc=sum)
total_births
subset = total_births[['William','Amy','Andrew','Ashley']]
#四个name趋势图
subset.plot(subplots=True,figsize=(12,15),grid=True,
            title='Numer of births per year')

table = top1000.pivot_table('prop',index=['year'],columns='sex',aggfunc=sum)
table.plot(title='sum of table1000.prop by year and sex',figsize=(10,5),
           yticks=np.linspace(0,1.0,11),xticks=range(1880,2020,10))

prop_cumsum = total_names.sort_index(by = 'prop',ascending=False).prop.cumsum()
prop_cumsum[:10]
prop_cumsum.searchsorted(0.5)
df = boys[boys.year == 1990 ]
df = df.sort_index(by='prop',ascending =False)

in1900 = df.sort_index(by='prop',ascending = False).prop.cumsum()
in1990 = in1900.searchsorted(0.5)+1

def get_quantile_count(group,q=0.5):
    group = group.sort_index(by='prop',ascending=False)
    return group.prop.cumsum().searchsorted(q)+1

diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.plot(title='Numblr of popular names in top 50%')

get_last_letter = lambda x : x[-1]
total_names['last'] = total_names.name.map(get_last_letter)
total_names[:5]
table = total_names.pivot_table('births',index = 'last' ,
                                columns=['sex','year'],aggfunc=sum)
subtable = table.reindex(columns=[1910,1960,2010],level='year')
subtable
subtable.sum()
letter_prop = subtable / subtable.sum().astype(float)
letter_prop
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1,figsize = (10,8))
letter_prop['M'].plot(kind='bar',rot=0,ax=axes[0],title='Male')
letter_prop['F'].plot(kind='bar',rot=0,ax=axes[1],title='Female',legend=False)

letter_prop = table /table.sum().astype(float)
dny_ts = letter_prop.ix[['d','n','y'],'M'].T
dny_ts.head()
dny_ts.plot()

all_names = total_names.name.unique()
all_names
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like
filtered = total_names[total_names.name.isin(lesley_like)]
filtered.groupby('name').births.sum()
table = filtered.pivot_table('births',index='year',columns = 'sex',
                             aggfunc = sum)
table.plot(style={'M':'k-','F':'k--'},figsize=(10,5))
