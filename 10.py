# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:28:29 2017

@author: chenshengkang
"""
#时间格式
from datetime import datetime
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
now = datetime.now()
now
now.year,now.month,now.day

delta = datetime(2011,1,7) - datetime(2008,6,24,8,15)
delta
delta.days
delta.seconds

from datetime import timedelta
start = datetime(2011,1,7)
start + timedelta(12)
start -2 * timedelta(12)


stamp = datetime(2011,1,3)
str(stamp)
stamp.strftime('%Y-%m-%d')

value = '2011-01-03'
datetime.strptime(value,'%Y-%m-%d')
datestrs = ['7/6/2011' ,'8/6/2011']

[datetime.strptime(x , '%m/%d/%Y') for x in datestrs]

from dateutil.parser import parse
parse('2011-01-03')
parse('Jan 31, 1997 10:45 PM')

parse('6/12/2011', dayfirst=True)

datestrs
pd.to_datetime(datestrs)
idx = pd.to_datetime(datestrs + [None])
idx
idx[2]
pd.isnull(idx)

from datetime import datetime
dates = [datetime(2011,1,2),datetime(2011,1,5),datetime(2011,1,7),
         datetime(2011,1,8),datetime(2011,1,10),datetime(2011,1,12)]

ts = Series(np.random.randn(6),index=dates)
ts
type(ts)
ts.index
ts + ts [::2]

ts.index.dtype
stamp = ts.index[0]
stamp

stamp = ts.index[2]
ts[stamp]

ts['1/10/2011']
ts['20110110']

longer_ts = Series(np.random.randn(1000), index = pd.date_range('1/1/2000',periods=1000))

longer_ts
longer_ts['2000']
longer_ts['2000-05']

ts[datetime(2011,1,7):]
ts
ts['20110106' : '20110111']
ts.truncate(after='20110109')

dates = pd.date_range('1/1/2000',periods=100,freq='W-WED')
long_df = DataFrame(np.random.randn(100,4),index=dates,
                    columns = ['Colorado','Texas','New York','Ohio'])
long_df.ix['5-2001']
dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000',
                      '1/3/2000'])
dup_ts = Series(np.arange(5),index=dates)
dup_ts
dup_ts.index.is_unique
dup_ts['20000103']
dup_ts['20000102']

grouped = dup_ts.groupby(level = 0)
grouped.mean()
grouped.count()

ts
rs = ts.resample('D')

index = pd.date_range('20120104','20120106')
index

pd.date_range(start = '20120104',periods = 20)
pd.date_range(end = '20120106',periods=20)
pd.date_range('1/1/2000','12/1/2000',freq='BM')
pd.date_range('5/2/2012 12:56:31',periods =5 , normalize =False)
pd.date_range('5/2/2012 12:56:31',periods =5 , normalize =True)
from pandas.tseries.offsets import Hour,Minute
hour = Hour(4)
hour
pd.date_range('1/1/2000','1/3/2000 23:59',freq='4h')

Hour(2) + Minute(30)

pd.date_range('1/1/2000',periods = 10, freq = '1h30min')

rng = pd.date_range('1/1/2012','9/1/2012',freq = 'WOM-3FRI')
list(rng)

ts = Series(np.random.randn(4),
            index = pd.date_range('1/1/2000',periods = 4 ,freq = 'M'))
ts
ts.shift(2)
ts.shift(-2)
ts/ts.shift(1)-1
ts.shift(2,freq = 'M')
ts.shift(3,freq='D')
ts.shift(1,freq = '3D')
ts.shift(1,freq='90T')
