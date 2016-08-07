#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yangyining'


import pandas as pd
import scipy as sp
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import time

#处理时间
def processOutTime(timeFrame):
    timeFrameCopy = timeFrame
    for timei in xrange(len(timeFrame)):
        timestr = str(timeFrame[timei])
        if 'year' in timestr:
            timeFrameCopy.iat[timei] = int(timestr[:2]) * 365
        elif 'week' in timestr:
            timeFrameCopy.iat[timei] = int(timestr[:2]) * 7
        elif 'month' in timestr:
            timeFrameCopy.iat[timei] = int(timestr[:2]) * 30
    return timeFrameCopy

def processDateTime(timeFrame):
    for timei in xrange(len(timeFrame)):
        timestr = str(timeFrame[timei])
        timeFrame.iat[timei] = datetime_timestamp(timestr)
    return timeFrame
#时间转换为unix时间戳
def datetime_timestamp(dt):
    # dt为字符串
    # 中间过程，一般都需要将字符串转化为时间数组
    time.strptime(dt, '%Y-%m-%d %H:%M:%S')
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=-1)
    # 将"2012-03-28 06:53:40"转化为时间戳
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s)

def processData(Data):
    drop = ['AgeuponOutcome', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'SexuponOutcome', 'Breed', 'Color']
    Data = rawData.drop(drop, 1)
    Data = pd.concat([Data, pd.get_dummies(rawData['SexuponOutcome'])], 1)
    Data = pd.concat([Data, pd.get_dummies(rawData['Breed'])], 1)
    Data = pd.concat([Data, pd.get_dummies(rawData['Color'])], 1)
    Data = pd.concat([Data, pd.get_dummies(rawData['OutcomeType'])], 1)

    Data = pd.concat([Data, processOutTime(rawData['AgeuponOutcome'])], 1)
    Data = pd.concat([Data, processDateTime(rawData['DateTime'])], 1)
    return Data

#生日和出走时间归一化
def nomorTime(timeFrame):
    # timeArray = np.array(timeFrame)
    # summary = np.sum(timeArray)
    # for i in xrange(len(timeFrame)):
    #     timeFrame.iat[i] = int(timeFrame.iat[i])/summary
    # return timeFrame
    timeFrame = timeFrame/(3600*24) - 15800-160
    return timeFrame

pro = preprocessing
rawData = pd.read_csv('train.csv')
rawTest = pd.read_csv('test.csv')

Data = processData(rawData)
Data['DateTime'] = nomorTime(Data['DateTime'])

print Data['DateTime']


