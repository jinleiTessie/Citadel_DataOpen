#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:39:52 2019

@author: jinlei
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv(r"..//..//data//testing//supermarket_data.csv")
#convert dataframe to list of list, remove nan
store_data=store_data.values.tolist()
store_data=[[bar for bar in foo if str(bar) != 'nan'] for foo in store_data]
#train model
association_rules = apriori(store_data, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)


#%%
for item in association_results:
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
