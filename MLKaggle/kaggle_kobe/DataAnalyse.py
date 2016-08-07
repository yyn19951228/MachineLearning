#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yangyining'

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#画投篮分布图
def scatter_plot_by_category(nona,feat):
    alpha = 0.1
    gs = nona.groupby(feat)
    #按照区域表色 the usage of rainbow
    cs = cm.rainbow(np.linspace(0,1,len(gs)))
    for g,c in zip(gs,cs):
        plt.scatter(g[1].loc_x,g[1].loc_y,color = c,alpha=alpha)




 #数据处理
rawData = pd.read_csv('data.csv')
SelectedDataLabel = ['loc_x','loc_y','shot_made_flag','season']
#直接这样就好了...太菜了
Data = rawData[SelectedDataLabel]
#去处无用数据
Data = Data[pd.notnull(Data['shot_made_flag'])]
#isin 找到label下特定值
Data2000 = Data[Data['season'].isin(['2000-01'])]
Data2001 = Data[Data['season'].isin(['2001-02'])]
Data2003 = Data[Data['season'].isin(['2002-03'])]

plt.subplot(131)
scatter_plot_by_category(Data2000,'shot_made_flag')
plt.subplot(132)
scatter_plot_by_category(Data2001,'shot_made_flag')
plt.subplot(133)
scatter_plot_by_category(Data2003,'shot_made_flag')
plt.show()