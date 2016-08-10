#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday Aug 10 2016

#纯碎是尝试一下svm和画图
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

# 导入数据
iris = datasets.load_iris()
# 只取前两列好画图。。
x = iris.data[:,:2]
 
y = iris.target

#一个步长
h = 0.02


#SVM正则化参数 就是软间隔的那个
C = 1.0 
#svc support vector classification
svc = svm.SVC(kernel='linear',C = C).fit(x,y)

# x_min就是第一列中最小值－1 以此类推
x_min,x_max = x[:,0].min() - 1,x[:,0].max() + 1
y_min,y_max = x[:,1].min() - 1,x[:,1].max() + 1

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
zz = np.c_[xx.ravel(),yy.ravel()]

Z = svc.predict(zz)

Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)

plt.scatter(x[:,0],x[:,1],c=y)


# plt.show()

print 'x' 
print x 
print 'y' 
print y
print 'xx'
print xx
print 'yy'
print yy
print 'zz'
print zz
print 'z' 
print Z