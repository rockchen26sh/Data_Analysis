# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:18:46 2017

@author: chenshengkang
"""

import numpy as np
#创建ndarray
data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)
arr1

#嵌套序列
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
arr2

#维度查询
arr1.shape
arr2.shape 
arr2.ndim #维度数量

arr1.dtype
arr2.dtype

np.zeros(10)
np.zeros(3,6)
np.empty(2,3,2)

np.arange(15)

#数据类型
arr1 = np.array([1,2,3],dtype = np.float64)
arr2 = np.array([1,2,3],dtype = np.int32)
arr1.dtype
arr2.dtype

#数据类型转换
arr = np.array([1,2,3,4,5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

#浮点数转换为int会截取掉小数部分
arr = np.array([3.7,-1.2,-2.6,0.5,12.9,10.1])
arr
arr.astype(np.int32)

int_array = np.arange(10)
calibers = np.array([.22,.270,.357,.380,.44,.50],dtype = np.float64)
int_array.astype(calibers.dtype) #使用calibers的dtype属性

empty_uint32 = np.empty(8,dtype='u4')
empty_uint32

#数组与标量之间的运算
arr = np.array([[1,2,3],[4,5,6]],dtype=np.float64)
arr
arr*arr
arr-arr
1/arr
arr**0.5

#索引和切片
arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr
#数组切片是原数组的试图，改变数组切片的值时原数组的对应值也会改变
arr_slice = arr[5:8]
arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr
