#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:17:32 2019

@author: jinlei
"""
import numpy as np

def mae(actual,prediction):
    errors = abs(prediction - actual)
    return round(np.mean(errors), 2)

def mape(actual,prediction):   #mean absolute percentage error (MAPE)
    errors = abs(prediction - actual)
    mape = 100 * (errors / actual)
    accuracy = 100 - np.mean(mape)
    return round(accuracy, 2)# in percentage
