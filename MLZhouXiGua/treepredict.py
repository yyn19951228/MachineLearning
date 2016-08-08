#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author   yyn19951228
# date  Sunday Aug 7 2016

# 离散值标签的CDRT 稍作改进就可以变为可分类连续值的决策树

my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

# p146
# 学习一下这种写法
def divideset(rows,column,value):
        split_function = None
        if isinstance(value,int) or isinstance(value,float):
                split_function =lambda row:row[column]>=value
        else:
                split_function= lambda row:row[column]==value

        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]

        return (set1,set2)

#p147
def uniquecounts(rows,column=4):
        # dict
        results = {}
        for row in rows:
                r = row[column]
                if r not in results:
                        results[r] = 0
                results[r]+= 1
        return results

#基尼不纯度
def giniimpurity(rows,column=4):
        total = len(rows)
        counts = uniquecounts(rows,column)
        imp = 0
        for k1 in counts:
                p1 = float(counts[k1])/total
                for k2 in counts:
                        if k1==k2:continue
                        p2 = float(counts[k2])/total
                        imp+=p1*p2
        return imp

#熵
def entropy(rows,column=4):
        from math import log
        log2 = lambda x:log(x)/log(2)
        results = uniquecounts(rows,column)
        # start
        ent = 0.0
        for r in results.keys():
                p = float(results[r])/len(rows)
                ent = ent-p*log2(p)
        return ent

#p150 根据熵来做CDRT
def buildtree(rows,column=4,scoref=entropy):
        if len(rows)==0:
                return decisionnode()
        current_score = scoref(rows,column)

        # 最大的增益
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(rows[0])-1
        for col in range(0,column_count):
                column_values = {}
                for row in rows:
                        column_values[row[col]]=1
                for value in column_values.keys():
                        (set1,set2) = divideset(rows,col,value)

                        p = float(len(set1))/len(rows)
                        gain = current_score-p*scoref(set1)-(1-p)*scoref(set2)
                        if gain>best_gain and len(set1)>0 and len(set2)>0:
                                best_gain = gain
                                best_criteria = (col,value)
                                best_sets = (set1,set2)
        #创建子分支
        if best_gain>0:
                trueBranch = buildtree(best_sets[0])
                falseBranch = buildtree(best_sets[1])
                return decisionnode(col = best_criteria[0],value = best_criteria[1],tb = trueBranch,fb = falseBranch)
        else:
                return decisionnode(results=uniquecounts(rows))


#p155 剪枝函数 
def prune(tree,mingain):
        if tree.tb.results== None:
                prine(tree.tb,mingain)
        if tree.fb.results== None:
                prune(tree.fb,mingain)


        if tree.tb.results!= None and tree.fb.results!= None:
                tb,fb = [],[]
                for v,c in tree.tb.results.items():
                        tb += [[v]]*c#可以看看后面的尝试 相当于是一种取巧吧
                for v,c in tree.fb.results.items():
                        fb += [[v]]*c

                delta = entropy(tb+fb)-(entropy(tb)+entropy(fb))/2
                if delta<mingain:
                        tree.tb,tree.fb=None,None
                        tree.results = uniquecounts(tb+fb)


def printtree(tree,indent=' '):
        if tree.results!=None:
                print str(tree.results)

        else:
                print str(tree.col)+':'+str(tree.value)+'?'
                print indent+'T->',
                printtree(tree.tb,indent+'  ')
                print indent+'F->',
                printtree(tree.fb,indent+'  ')


#p158 用方差来对数值型结果进行处理
def variance(rows):
        if len(rows) == 0 : return 0
        data = [float(row[len(row)-1]) for row in rows]
        mean = sum(data)/len(data)
        variance = sum([(d-mean)**2 for d in data]) len(data)
        return variance



# CART Classification and Regression Trees
class decisionnode:
        def __init__(self,col=-1,value = None,results=None,tb = None,fb = None):
                self.col = col
                self.value = value
                self.results = results
                self.tb  = tb
                self.fb = fb


# printtree(buildtree(my_data))




# a = {'a':1,'b':2,'c':3}
# b = []
# for v,c in a.items():
#         b += [[v]]*c
# print b













