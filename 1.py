#!/usr/bin/python
# coding:utf-8

path = 'usagov_bitly_data2012-03-16-1331923249.txt'
print(open(path).readline())

import json
path = 'usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]
print records[0]  #json trans tupe

print records[0]['tz']  #加入键值

#对时区进行计时
time_zones = [rec['tz'] for rec in records if 'tz' in rec] #时区信息文本
print time_zones

#对时区进行计数,方法1
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

#对时区进行计数，方法2
from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int) #所有的值均会被初始化为0
    for x in sequence:
        counts[x] += 1
    return counts

counts = get_counts(time_zones)
counts2 = get_counts2(time_zones)
print counts
print counts2

#得到前10位的时区及计数值
def top_counts(count_dict,n=10):
    value_key_pairs = [(count,tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

c = top_counts(counts)
c.sort(reverse=True) #降序排列
print(c)

#使用collections.Counter实现
from collections import Counter
couts = Counter(time_zones)
print(couts.most_common(10)) #最多的10个

#使用pandas对时区进行计数
#将数据放入数据框
from pandas import DataFrame,Series
import pandas as pd
import numpy as np
frame = DataFrame(records)


print(frame['tz'][:10])

tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])

#未知和缺失值处理
clean_tz = frame['tz'].fillna('Missing') #筛选NA的填写missing
clean_tz[clean_tz == ''] = 'Unknown' #none 填写Unknown
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])

import matplotlib.pyplot as plt
plt.show(tz_counts[:10].plot(kind='barh',rot=0))


#a字段处理,字符串分离
print(frame['a'][:5])
results = Series([x.split()[0] for x in frame.a.dropna()])
print results[:5]

print(results.value_counts()[:8])

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not windows')
by_tz_os = cframe.groupby(['tz',operating_system]) #分类汇总函数groupby

agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])

indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]

p = count_subset.plot(kind = 'barh',stacked=True)
plt.show(p)
