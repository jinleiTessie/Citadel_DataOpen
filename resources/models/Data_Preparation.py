#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:43:38 2019

@author: jinlei
"""

import pandas as pd
def load_data(path):
    return pd.read_excel(path)

df=pd.read_excel(r"..//data//testing//Color.xlsx")
print (df.head())
print (df.shape)
print (df.describe())
dummy_df=pd.get_dummies(df)
print (dummy_df)
dummy_df=dummy_df.drop(["Color_Black","Color_Yellow"],axis=1)#drop columns
print (dummy_df)
dummy_df=dummy_df.drop([0,2],axis=0)#drop rows
print (dummy_df)

