# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:02:40 2017

@author: cck{28488747}
"""

import pandas as pd
import numpy as np

#读取数据
unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('movielens/users.dat',
                      sep='::',
                      header=None,
                      names=unames)

rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('movielens/ratings.dat',
                        sep='::',
                        header=None,
                        names=rnames)

mnames = ['movie_id','title','genres']
movies = pd.read_table('movielens/movies.dat',
                       sep='::',
                       header=None,
                       names=mnames)
#数据整合
data = pd.merge(pd.merge(ratings,users),movies)

#数据透视
mean_ratings = pd.pivot_table(data,
                              index='title',
                              columns='gender',
                              values='rating',
                              aggfunc=np.mean)

#计数
ratings_by_title = data.groupby('title').size()

active_titles = ratings_by_title.index[ratings_by_title>=250]

mean_ratings = mean_ratings.ix[active_titles]

top_female_ratings = mean_ratings.sort_index(by='F',
                                             ascending=False)

#男女评分差
mean_ratings['diff'] = mean_ratings['M']-mean_ratings['F']
#男性欢迎和女性欢迎电影
mean_ratings.sort_index(by='diff',ascending=False)
#方差，查看评分差最大的电影
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]
rating_std_by_title.order(ascending=False)[:10]
