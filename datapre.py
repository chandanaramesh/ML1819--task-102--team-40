#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:19:18 2018

@author: rameshsu
"""
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import svm
from sklearn import metrics
       
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

#Create a svm Classifier
#clf = svm.SVC(kernel='linear') # Linear Kernel

clf = svm.SVC(kernel='rbf', random_state=0, gamma=0.01, C=1) # Radial Model

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = clf.predict(X_test)

accuracy=metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy:",accuracy)
#print("Precision:",metrics.precision_score(Y_test, Y_pred))
#print("Recall:",metrics.recall_score(Y_test, Y_pred))

plt.style.use('seaborn-whitegrid')
#plt.set_xlim(0, 1.5)
plt.title('Predicted Values')
plt.xlabel('Appliance')
plt.ylabel('Value')

plt.plot(Y_test,'g^')
plt.plot(Y_pred,'r^')
plt.savefig('Graph_PredictedValue_Gamma.png')
plt.show()

#plt.title('Test Values')
#plt.xlabel('Appliance Index')
#plt.ylabel('Value')
#plt.plot(Y_test,color='red')
#plt.show()
#plt.savefig('Graph_TestValue.png')
