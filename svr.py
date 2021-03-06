import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime as dt

dataset = pd.read_csv('energydata.csv', delimiter=',')

#Convert datetime to ordinal value
dataset.date.dtype
dataset['date']= pd.to_datetime(dataset['date'])
dataset['date'] = dataset['date'].map(dt.toordinal)
dataset.Appliances.dtype

#Assigning dependent and independent variables
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Replace missing values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:28])
X[:, 1:28] = imputer.transform(X[:, 1:28])

#Split the data between the Training Data and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.35, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm_X = MinMaxScaler()
X_train = mm_X.fit_transform(X_train)
X_test = mm_X.transform(X_test)

#SVR Algorithm
from sklearn.svm import SVR
regressor2=SVR(kernel='rbf')
regressor2.fit(X_train, Y_train)
y_pred2 = regressor2.predict(X_test)
rms2 = sqrt(mean_squared_error(Y_test, y_pred2))
rsqrd2=r2_score(Y_test, y_pred2)
