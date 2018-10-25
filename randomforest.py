import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime as dt

dataset = pd.read_csv('energydata.csv', delimiter=',')
#dataset['Appliances']= dataset['Appliances'].astype('category')

dataset.date.dtype
dataset['date']= pd.to_datetime(dataset['date'])
dataset['date'] = dataset['date'].map(dt.toordinal)
dataset.Appliances.dtype
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

d1=dataset[0:0]
i=0

for i in range(19735):
    if dataset['Appliances'][i] < 50:
        d1[i]=dataset[i]

dfil = dataset[(dataset.Appliances > 50)]
print(dfil)

X1= X
X1=pd.DataFrame(dataset)
#X1=X1.drop("T9","Visibility","rv1","rv2", axis=0)
X1.columns= dataset.columns
X1.drop(X1.columns[[18,26,27]], axis=1, inplace=True)
X = X1.iloc[:, :-1].values

#X.drop(X.columns[[18,26,27]], axis=1, inplace=True)

"""
Y=Y.reshape(-1,1)
Y = pd.DataFrame(Y)
Y.columns = ['Appliances']
Y['Appliances']= Y['Appliances'].astype('category')
Y.Appliances.dtype
"""
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


regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, Y_train)  
y_pred = regressor.predict(X_test)

print(regressor.feature_importances_)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(Y_test, y_pred))
