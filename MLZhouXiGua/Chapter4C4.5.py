#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday Aug 7 2016


class C45Classifier():
	def Ent(self):
		temp = []
		Ta = []
		T = [[self.X_train[i][j] for i in range(0,len(self.X_train))]for j in range(0,len(self.X_train[0]))]
		for i in range(0,len(T)):
			T[i].sort()
		# 计算候选分裂值	
		for i in range(0,len(T)):
			for ii in range(0,len(T[i])-2):
				temp.append((T[i][ii] + T[i][ii+1])/2)
			Ta.append(temp)
			temp = []

		return Ta

	def fit(self,X_train,Y_train):
		self.X_train = X_train
		self.Y_train = Y_train
		

	def predict(self,X_test):
		pass

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import  train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= .5)


my_classifier = C45Classifier()
my_classifier.fit(X_train, Y_train)

print my_classifier.Ent()



# predicitions = my_classifier.predict(X_test)

# from sklearn.metrics import accuracy_score
# print accuracy_score(Y_test,predicitions)


