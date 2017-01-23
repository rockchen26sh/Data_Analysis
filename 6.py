# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:51:49 2017

@author: chenshengkang
"""
import pandas as pd
from pandas import DataFrame, Series
df = pd.read_csv('ch06/ex1.csv')
df
pd.read_table('ch06/ex1.csv',sep=',')

pd.read_csv('ch06/ex2.csv',header = None)
names = ['a','b','c','d','message']
pd.read_csv('ch06/ex2.csv', names = names,index_col='message')
pd.read_csv('ch06/ex2.csv', names = names).set_index('message')


pd.read_csv('ch06/csv_mindex.csv')
pd.read_csv('ch06/csv_mindex.csv',index_col=['key1','key2'])
pd.read_csv('ch06/csv_mindex.csv').set_index(['key1','key2'])

list(open('ch06/ex3.txt'))
result = pd.read_table('ch06/ex3.txt',sep='\s+')
result

list(open('ch06/ex4.csv'))
pd.read_csv('ch06/ex4.csv',skiprows=[0,2,3])

list(open('ch06/ex5.csv'))
result = pd.read_csv('ch06/ex5.csv',na_values=['NULL'])
result

sentinels = {'message':['foo','NA'], 'something':['two']}
pd.read_csv('ch06/ex5.csv',na_values=sentinels) #指定值填充为na

result = pd.read_csv('ch06/ex6.csv')
result
pd.read_csv('ch06/ex6.csv',nrows=5)
chunker = pd.read_csv('ch06/ex6.csv',chunksize=1000)
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(),fill_value=0)

tot = tot.order(ascending=False)
tot

data = pd.read_csv('ch06/ex5.csv')
data.to_csv('ch06/out.csv')
list(open('ch06/out.csv'))

import csv
f = open('ch06/ex7.csv')
reader = csv.reader(f)
reader
for line in reader:
    print (line)

line = list(csv.reader(open('ch06/ex7.csv')))
header,values = line[0] , line[1:]
data_dict = {h:v for h , v in zip(header,zip(*values))}
data_dict

#json读取
obj = """{"name":"wes","places_lived":["United States","Spain","Germar"],
"per":null,"siblings":[{"name":"Scott","age":25,"pet":"Zuko"},
{"name":"Katie","age":33,"pet":"Cisco"}]}"""
obj
import json
result = json.loads(obj)
result
asjson = json.dumps(result)
asjson
siblings = DataFrame(result['siblings'],columns=['name','age'])
siblings
#XML和HTML
from lxml.html import parse
from urllib.request import urlopen
from urllib.request import requests
parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()
links = doc.findall('.//a')
lnk = links[28]
lnk.get('href')
lnk.text_content()

urls = [lnk.get('href') for lnk in doc.findall('.//a')]
urls

tables = doc.findall('.//table')
calls = tables
rows = calls.findall('.//tr')


