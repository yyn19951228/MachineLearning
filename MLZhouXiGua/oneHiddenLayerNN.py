#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday Aug 9 2016

#单隐藏层NN
import numpy as np

#sigmoid function
def nonlin(x,deriv=False):
	if (deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

#input dataset
x = np.array([	
	[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]
	])

#output dataset
y = np.array([[1,0,1,1]]).T
print y.shape
np.random.seed(2)

#学习速率
ita = 1
#initialize weights randomly with mean 0
#连接权值
syn0 = 2*np.random.random((3,5))-1
syn1 = 2*np.random.random((5,1))-1

for iter in xrange(100000):
	#forward propagation
	#其中l0为输入层 l1为隐藏层 l2为输出层
	l0 = x
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))

	l2_error = y - l2

	if iter % 10000 == 0:
		print "error:" + str(np.mean(np.abs(l2_error)))

	#可以看出BP的感觉 逆向传播 从后向前
	l2_delta = l2_error * nonlin(l2,deriv=True)

	l1_error = np.dot(l2_delta,syn1.T)

	l1_delta = l1_error * nonlin(l1,deriv= True)
	

	syn0 += np.dot(l0.T,l1_delta)
	syn1 += np.dot(l1.T,l2_delta)

print "Output After Training:"
print l2
