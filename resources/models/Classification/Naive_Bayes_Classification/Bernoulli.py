#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:12:55 2019

@author: jinlei
"""
#features are True(1)/False(0) counts

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
#input
training_data=pd.DataFrame({"Walks_like_duck":[1, 1, 0],"Talks_like_duck":[0, 0, 1],"Is_small":[1, 0, 0],"class":['Duck', 'Not a Duck', 'Not a Duck']})
testing_data=pd.DataFrame({"Walks_like_duck":[1,0],"Talks_like_duck":[0,0],"Is_small":[1,0]})
input_X = training_data.iloc[:,:-1]
input_y = training_data.iloc[:,-1]

Bernoulli_Model = BernoulliNB()
Bernoulli_Model.fit(input_X, input_y)

#prediction on new data
prediction_result=Bernoulli_Model.predict(testing_data)
print (prediction_result)
