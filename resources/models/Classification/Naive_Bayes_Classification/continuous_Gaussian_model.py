#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:09:46 2019

@author: jinlei
"""
###Naive_Bayes_Classification:
#assumption: all features are independent, usually not true.
#algorithm:算一个数据点为各个class的可能性、选其为最高probability的class
#pros: need less training data, speed is fast
#cons: made many assumptions
#####

####Gaussian Model (Continuous)
#assumption: features follow a normal distribution

from sklearn.naive_bayes import GaussianNB
import pandas as pd

#load input
data=pd.read_excel(r"..//..//..//data//testing//Color.xlsx")
input_X=data[["Red","Green","Blue"]]
input_y=data[["Color"]]

#Training
Gaussian_model = GaussianNB()
Gaussian_model.fit(input_X, input_y)
#prediction on new data
testing_data=pd.DataFrame({"Red":[1,0],"Green":[0,0],"Blue":[1,0]})
prediction_result=Gaussian_model.predict(testing_data)
print (prediction_result)
