#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yangyining'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
import time
import scipy as sp


#画投篮分布图
def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = nona.groupby(feat)
    #按照区域表色 the usage of rainbow
    cs = cm.rainbow(np.linspace(0,1,len(gs)))
    for g,c in zip(gs,cs):
        plt.scatter(g[1].loc_x,g[1].loc_y,color = c,alpha=alpha)

#损失函数
def logloss(act,pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon,pred)
    pred = sp.minimum(1-epsilon,pred)
    #实际上我觉得这个式子就是机器学习课程中的cost Function
    #sum(act*log(pred) + (1-a)*log(1-p))
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


# import data
filename = "data.csv"
raw = pd.read_csv(filename)
nona =  raw[pd.notnull(raw['shot_made_flag'])]

#得到距离
raw['dist'] = np.sqrt(raw['loc_x']*raw['loc_x'] + raw['loc_y']*raw['loc_y'])


#得到比赛剩余时间
raw['remaining_time'] = raw['minutes_remaining']*60+raw['seconds_remaining']

action_type = nona.action_type.unique()
combined_shot_type = nona.combined_shot_type.unique()
shot_type = nona.shot_type.unique()
season = nona.season.unique()

#去掉多余的变量
drops = ['shot_id','team_id','team_name','shot_zone_area','shot_zone_range','shot_zone_basic',
         'matchup','lon','lat','seconds_remaining','minutes_remaining','shot_distance','loc_x','loc_y',
         'game_id','game_date']
for drop in drops:
    raw = raw.drop(drop,1)

#把分类变量变为名义变量
categorical_vara = ['action_type','combined_shot_type','shot_type','opponent','period','season']
for var in categorical_vara:
    raw = pd.concat([raw,pd.get_dummies(raw[var],prefix=var)],1)
    raw = raw.drop(var,1)

print raw

#seperate data
df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag',1)
#separate df into explanatory and response variables
train = df.drop('shot_made_flag',1)
train_y = df['shot_made_flag']


#building model!!!!!
#Random Forest

#find the best n_estimators for RandomForestClassifer


print 'Finding best n_estimators for RandomForestClassfier...'
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(1,5,num=20).astype(int)
#range_n = [1 10 100]
for n in range_n:
    print("the number of tress: {0}".format(n))
    t1 = time.time()

    rfc_score = 0
    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(len(train),n_folds=10,shuffle=True):
        rfc.fit(train.iloc[train_k],train_y.iloc[train_k])
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k],pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()

    print("Done processing {0} trees ({1:.3f}sec)".format(n,t2-t1))
print(best_n,min_score)

best_n = 1000

#find best max_depth for RandomForestClassifier
print 'Find best max_depth for RandomForestClassifier..'
min_score = 100000
best_m = 0
scores_m = []
range_m = np.logspace(0,2,num=20).astype(int)
for m in range_m:
    print("the max depth: {0}".format(m))
    t1 = time.time()

    rfc_score = 0
    rfc = RandomForestClassifier(max_depth=m,n_estimators=best_n)
    for train_k,test_k in KFold(len(train),n_folds=10,shuffle=True):
        rfc.fit(train.iloc[train_k],train_y.iloc[train_k])
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k],pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m

    t2 = time.time()
    print('Done Processing {0} trees ({1:.3f}sec)'.format(m,t2-t1))
print (best_m,min_score)


#building a final model

best_n = 100
best_m = 30

model = RandomForestClassifier(n_estimators=best_n,max_depth=best_m)
model.fit(train,train_y)
pred = model.predict_proba(submission)

#predicted shot_made_flag is written to a csv file
sub = pd.read_csv('sample_submission.csv')
sub['shot_made_flag'] = pred
sub.to_csv('reaL_submission.csv',index=False)




'''
#plot shot_zone_area
plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')
#plot_zone_basic
plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')
#plot_zone_range
plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')
plt.show()
'''
'''#得到经纬度和角度
loc_x_zero = raw['loc_x'] == 0
raw['angle'] = np.array([0]*len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi / 2'''
'''
alpha = 0.02
plt.figure(figsize=(10,10))

# loc_x and loc_y
plt.subplot(121)
plt.scatter(nona.loc_x, nona.loc_y, color='blue', alpha=alpha)
plt.title('loc_x and loc_y')

# lat and lon
plt.subplot(122)
plt.scatter(nona.lon, nona.lat, color='green', alpha=alpha)
plt.title('lat and lon')
'''
