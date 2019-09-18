#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:05:19 2019

@author: jinlei
"""
#features are discrete counts
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

data=pd.read_excel("..//..//..//data//testing//Color.xlsx")
input_X=data[["Red","Green","Blue"]]
input_y=data[["Color"]]

#Training
Multinomial_Model = MultinomialNB()
Multinomial_Model.fit(input_X, input_y)

#prediction on new data
testing_data=pd.DataFrame({"Red":[1,0],"Green":[0,0],"Blue":[1,0]})
prediction_result=Multinomial_Model.predict(testing_data)
print (prediction_result)
