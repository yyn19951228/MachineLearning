#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yangyining'


import pandas as pd
import xgboost as xg
import scipy as sp
import numpy as np
import xgboost as xgb

rawData = pd.read_csv('data.csv')
SelectedDataLabel = ['loc_x','loc_y','shot_made_flag','action_type','combined_shot_type','game_event_id']
Data = rawData[SelectedDataLabel]

Data['dist'] = np.sqrt(Data['loc_x']*Data['loc_x']+Data['loc_y']*Data['loc_y'])

Data = Data[pd.notnull(Data['shot_made_flag'])]

Data = Data.drop('loc_x',1)
Data = Data.drop('loc_y',1)

