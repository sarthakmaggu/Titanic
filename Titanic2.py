#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:57:00 2020

@author: maggu
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('train.csv')
train.dropna(inplace=True)
target = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Fare']
X = train[features]
y = train[target]
model = RandomForestRegressor()
model.fit(X, y)
model.score(X, y)
pickle.dump(model, open('model.pkl', 'wb'))