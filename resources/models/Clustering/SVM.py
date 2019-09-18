#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:05:19 2019

@author: jinlei
"""
#The pros to SVM’s:
#pros: Effective for higher dimension. Best classifier when data points are separable. Saves memory.
#cons: training times are long for large data set. Perform bad for non-separable data points.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_breast_cancer

dataCancer = load_breast_cancer()
data = dataCancer.data[:, 0:2]
target = dataCancer.target

#C: Penalty parameter of the error term
model = svm.SVC(kernel = 'rbf', C = 10000)#kernel:'linear','poly','rbf'
model.fit(data, target)

# plot
plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.prism)
# Creates the axis bounds for the grid
axis = plt.gca()
x_limit = axis.get_xlim()
y_limit = axis.get_ylim()

# Creates a grid to evaluate model
x = np.linspace(x_limit[0], x_limit[1], 50)
y = np.linspace(y_limit[0], y_limit[1], 50)
X, Y = np.meshgrid(x, y)
xy = np.c_[X.ravel(), Y.ravel()]

# Creates the decision line for the data points, use model.predict if you are classifying more than two 
decision_line = model.decision_function(xy).reshape(Y.shape)

# Plot the decision line and the margins
axis.contour(X, Y,  decision_line, colors = 'k',  levels=[0], linestyles=['-'])
# Shows the support vectors that determine the desision line
axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, 
             linewidth=1, facecolors='none', edgecolors='b')
plt.show()

