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

data = DataFrame({'food':['bacon','pulled pork','bacon','Pastrami','corned beef','Bacon','pastrami','honey ham','nova lox'],
                  'ounces':[4,3,12,6,7.5,8,3,5,6]})
data
meat_to_animal = {'bacon':'pig','pulled pork':'pig','pastrami':'cow',
                  'corned beef':'cow','honey ham':'pig','nova lox':'salmon'}
meat_to_animal
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data
data['food'].map(lambda x:meat_to_animal[x.lower()])

data = DataFrame(np.arange(12).reshape((3,4)),
                 index = ['Ohio','Colorado','New York'],
                 columns = ['one','two','three','four'])

data.index = data.index.map(str.upper)
data
data.columns = data.columns.map(str.upper)
data
data.rename(index = {'OHIO':'INDIANA'},columns = {'THREE':'peekaboo'})
_ = data.rename(index = {'OHIO':'INDIANA'},inplace = True)
data

ages = [20,22,25,27,21,23,37,31,61,45,41,32]
bins = [18,25,35,60,100]
cats = pd.cut(ages,bins)
cats
cats.labels
cats.levels
pd.value_counts(cats)
pd.cut(ages,[18,26,36,61,100],right=False)

groupnames = ['Youth','YoungAdult','MiddleAged','Senior']
cats2 = pd.cut(ages,bins,labels=groupnames)
pd.value_counts(cats2)
cats2

data = np.random.rand(20)
data
pd.cut(data,4,precision=2)

data = np.random.randn(1000)
cats = pd.qcut(data,4)
cats
pd.value_counts(cats)
pd.qcut(data,[0,0.1,0.5,0.9,1.]).value_counts()

np.random.seed(12345)
data = DataFrame(np.random.randn(1000,4))
data.describe()
data
col = data[3]
col[np.abs(col) > 3]
col[col > 3 | col < -3]
data[(np.abs(data) > 3).any(1)]
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()

df = DataFrame(np.arange(5 * 4).reshape(5,4))
df
sampler = np.random.permutation(5)
sampler
df.index = sampler
df.take(sampler)
df.take(np.random.permutation(len(df))[:3])
df.take([0,1])

bag = np.array([5,7,-1,6,4])
sampler = np.random.randint(0,len(bag),size = 10)
sampler
draws = bag.take(sampler)
draws

d = DataFrame({'key':['b','b','a','c','a','b'],
               'data':range(6)})
d
pd.get_dummies(d['key'])

dummies = pd.get_dummies(d['key'],prefix='key')
dummies
df_with_dummy = d[['data1']].join(dummies)
d['data1'] = d.index
d
d[['data1']].join(dummies)

mnames = ['movie_id','title','genres']
movies = pd.read_table('Data_Analysis/ch02/movielens/movies.dat',sep = '::',header = None,
                       names = mnames )
movies[:10]
genre_iter = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iter))
dummies = DataFrame(np.zeros((len(movies),len(genres))),columns = genres)
dummies

for i , gen in enumerate(movies.genres):
    dummies.ix[i,gen.split('|')] = 1
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.ix[0]

values = np.random.rand(10)
values
bins = [0,0.2,0.4,0.6,0.8,1]
pd.get_dummies(pd.cut(values,bins))


val = 'a,b, guido'
val.split(',')
pieces = [x.strip() for x in val.split(',')]
pieces
first ,second, third = pieces
first +'::' + second + ':: ' + third
'guido' in val
val.index(',')
val.find(':')
val.count(',')
val.replace(',',':: ')
val.replace(',','')

text = "foo bar\t baz   \tqux"
import re
re.split('\s+',text)
regex = re.compile('\s+')
regex.split(text)
regex.findall(text)

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern,flags = re.IGNORECASE)
regex.findall(text)
m = regex.search(text)
m
text[m.start():m.end()]
print (regex.match(text))
print (regex.sub('REDACTED',text))
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern,flags = re.IGNORECASE)



m = regex.match('wesm@bright.net')
m.groups()
regex.findall(text)
print (regex.sub(r'Username:\1, Domain: \2,Suffix: \3',text))
regex = re.compile(r"""
(? P<username>[A-Z0-9._%+-]+)
@
 (? P<domain>[A-Z0-9.-]+)
\.
 (? P<suffix>[A-Z]{2,4})""",flags = re.IGNORECASE|re.VERBOSE)

import json
db = json.load(open('Data_Analysis/ch07/foods-2011-10-03.json'))
len(db)
db[0].keys()
nutrients = DataFrame(db[0]['nutrients'])
nutrients
info_keys = ['description','group','id','manufacturer']
info = DataFrame(db,columns = info_keys)
info[:5]

info.group.value_counts()
db[:5]
nutrients = []
for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)
nutrients[:5]
nutrients = pd.concat(nutrients,ignore_index=True)
nutrients = nutrients.drop_duplicates()
col_mapping = {'description':'food','group':'fgroup'}
info = info.rename(columns=col_mapping,copy=False)
info[:5]
col_mapping = {'description':'nutrient','group':'nutgroup'}
nutrients = nutrients.rename(columns = col_mapping,copy=False)
nutrients[:5]
ndata = pd.merge(info,nutrients,on = 'id',how='outer')
ndata[:5]
result = ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)
result
result.order().plot(kind = 'barh',figsize=[8,200])
by_nutrient = ndata.groupby(['nutgroup','nutrient'])

get_maximum = lambda x: x.xs(x.value.idxmax())
get_minimum = lambda x: x.xs(x.value.idxmin())
max_foods = by_nutrient.apply(get_maximum)[['value','food']]
max_foods.food = max_foods.food.str[:50]
max_foods.ix['Amino Acids']['food']
