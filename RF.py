# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:00:21 2018

@author: Colonel Grizzly Bear
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv(r'energydata.csv', delimiter=',')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:30])
X[:, 1:30] = imputer.transform(X[:, 1:28])

#Split the data between the Training Data and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm_X = MinMaxScaler()
X_train = mm_X.fit_transform(X_train)
X_test = mm_X.transform(X_test)


regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, Y_train)  
y_pred = regressor.predict(X_test) 
