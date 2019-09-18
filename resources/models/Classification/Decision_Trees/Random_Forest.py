#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:33:03 2019

@author: jinlei
"""
import pandas as pd
import numpy as np
import models.result_evaluation as evaluate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

input_df=pd.read_csv(r"..//..//..//data//testing//temps.csv")
input_df = pd.get_dummies(input_df)
feature_list=list(set(input_df.columns)-set(["actual"]))
train_features, test_features, train_labels, test_labels = train_test_split(input_df[feature_list], input_df[["actual"]], test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#######Training
#n_estimators:1000 number of trees in the forest
#random_state: seed for random number generator
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)
#prediction
test_labels["predictions"] = rf.predict(test_features)
print ('Mean Absolute Error:', evaluate.mae(test_labels["predictions"],test_labels["actual"]))
