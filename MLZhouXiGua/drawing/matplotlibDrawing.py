#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday August 10 2016


import matplotlib.pyplot as plt 
import numpy as np 

#最简单的 正弦曲线
# x = np.arange(0,5,0.1)
# y = np.sin(x)
# plt.plot(x,y)


#黑色散点图 颜色标示和Matlab差不多
# x = np.array([[1,2],[3,4],[5,6]])
# plt.plot(x[:,0],x[:,1],'ko')


#直方图
# a = np.random.normal(5.0,3.0,1000)
# plt.hist(a,100,color='green',alpha=0.4)

#2D直方图（类似热成像图）
# x = np.random.random([1000,2]) * 10
# plt.hist2d(x[:,0],x[:,1],bins=20)


# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.show
# 还是直接查看文档比较方便


plt.show()