#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:42:54 2019

@author: jinlei
"""
#how to choose K:
#higher K: will ignore outliers to the data. K cannot be too high, not able to classify the data
#low K: give more weight to outliers, k needs to be relatively small.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# get data
from sklearn.datasets import load_breast_cancer
dataCancer = load_breast_cancer()
tranining_put=pd.DataFrame(dataCancer.data[:, 0:2], columns=dataCancer.feature_names[0:2])
target = dataCancer.target
#train model
model = KNeighborsClassifier(n_neighbors = 9, algorithm = 'auto')
model.fit(tranining_put, target)


# plots
plt.scatter(tranining_put.iloc[:, 0], tranining_put.iloc[:, 1], c=target, s=30, cmap=plt.cm.prism)
# Creates the axis bounds for the grid
axis = plt.gca()
x_limit = axis.get_xlim()
y_limit = axis.get_ylim()
# Creates a grid to evaluate model
x = np.linspace(x_limit[0], x_limit[1])
y = np.linspace(y_limit[0], y_limit[1])
X, Y = np.meshgrid(x, y)
xy = np.c_[X.ravel(), Y.ravel()]
# Creates the line that will separate the data
boundary = model.predict(xy)
boundary = boundary.reshape(X.shape)
# Plot the decision boundary
axis.contour(X, Y,  boundary, colors = 'k')         
# Shows the graph
plt.show()
