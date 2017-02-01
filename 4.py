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

#高维数组的索引
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2]
arr2d[0][2]
arr2d[0,2]

arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d
arr3d[0]
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d

#切片语法
arr[1:6]
arr2d
arr2d[:2]
arr2d[:2,1:]

#布尔索引
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
names
data
names == 'Bob'
data[names == 'Bob']
data[names == 'Bob' ,2:]
data[names == 'Bob' ,3]

names != 'Bob'
data[names != 'Bob']
data[names != 'Bob' ,2:]
data[names != 'Bob' ,3]

data[data<0] = 0
data

data[names != 'Joe'] = 7
data

#花式索引
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr
arr[[4,3,0,6]] #返回第4，3，0，6行
arr[[-3,-5,-7]] #返回 5 ,4,1行

#数组转置
arr = np.arange(15).reshape((3,5))
arr
arr.T
arr = np.random.randn(6,3)
np.dot(arr.T,arr)
arr = np.arange(16).reshape((2,2,4))
arr
arr.transpose((1,0,2))
arr.swapaxes(1,2)

#通用函数
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x= np.random.randn(8)
y= np.random.randn(8)
x
y
np.maximum(x,y)

arr = np.random.randn(7)*5
arr
np.modf(arr)


points = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(points,points)
xs
ys
import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 +ys ** 2)
z
plt.imshow(z,cmap=plt.cm.gray)
plt.colorbar()
plt.title("image plot of $\sqrt(x^2+y^2)$ for a grid of values")

xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])

result = [(x if c else y) for x, y, c in zip(xarr,yarr,cond)]
result
result = np.where(cond,xarr,yarr)
result
arr = np.random.randn(4,4)
arr.dtype
np.where(arr>0 ,2,-2)
np.where(arr>0 ,2,arr)

