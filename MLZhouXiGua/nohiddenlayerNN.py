#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday Aug 9 2016

#无隐藏层NN
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
y = np.array([[0,0,1,1]]).T

np.random.seed(2)

#学习速率
ita = 1
#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1))-1

for iter in xrange(10000):
	#forward propagation
	l0 = x
	l1 = nonlin(np.dot(l0,syn0))

	# 误差
	l1_error = y-l1
	#基于误差的修正值
	l1_delta = ita* l1_error * nonlin(l1,True)

	syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1