#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday Aug 9 2016

#多隐藏层NN
import numpy as np
from sklearn import datasets

#sigmoid function
def nonlin(x,deriv=False):
	if (deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

def processData(testData):
	testData = np.array([testData])
	return testData.T

class NN():
	def __init__ (self, X,Y,n,hn):
		self.X = X
		self.Y = Y
		self.n = n
		self.hn = hn
		

	def fit(self):
		np.random.seed(2)
		#学习速率
		ita = 1
		#initialize weights randomly with mean 0
		#连接权值
		syn0 = 4*np.random.random((len(self.X[0]),self.hn))-2
		syn1 = 4*np.random.random((self.hn,len(self.Y[0])))-2

		for iter in xrange(self.n):
			#forward propagation
			#其中l0为输入层 l1为隐藏层 l2为输出层
			l0 = self.X
			l1 = nonlin(np.dot(l0,syn0))
			l2 = nonlin(np.dot(l1,syn1))

			l2_error = self.Y - l2

			if iter % 10000 == 0:
				print "error:" + str(np.mean(np.abs(l2_error)))

			#可以看出BP的感觉 逆向传播 从后向前
			l2_delta = ita*l2_error * nonlin(l2,deriv=True)

			l1_error = np.dot(l2_delta,syn1.T)

			l1_delta = ita* l1_error * nonlin(l1,deriv= True)
			

			syn0 += np.dot(l0.T,l1_delta)
			syn1 += np.dot(l1.T,l2_delta)

		self.syn0 = syn0
		self.syn1 = syn1
		print l2
		

	def predict(self,X_test):
		testResult = np.dot(X_test,self.syn0)
		
		testResult = np.dot(testResult,self.syn1)
		
		return testResult/300






iris = datasets.load_iris()
X = iris.data
Y = iris.target

from sklearn.cross_validation import  train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= .3)
Y_test = processData(Y_test)
Y_train = processData(Y_train)




classifier = NN(X_train,Y_train,60000,10)
classifier.fit()



# print Y_test


result = classifier.predict(X_test)
print result




