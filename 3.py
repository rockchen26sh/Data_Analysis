# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:51:14 2017

@author: cck{28488747}
"""
import pandas as pd
import numpy as np
#读取1880出生数据
names_1880 = pd.read_csv("names/yob1880.txt",
                         header=None,
                         names = ['name','sex','births'])
#按性别分组计算出生人数
names_1880.groupby('sex').births.sum()

#批量读取
years = range(1880,2011)
col = ['name','sex','births']
total_names = []

for year in years:
    path = "names/yob" + str(year) + '.txt'
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
subset = total_births[['Amanda','Amy','Andrew','Ashley']]
subset.plot(subplots=False,figsize=(12,5),grid=True,
            title='Numer of births per year')


