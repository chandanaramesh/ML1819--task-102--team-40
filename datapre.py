# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:30:27 2018

@author: rosha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

   
        
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime as dt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

dataset = pd.read_csv('energydata.csv', delimiter=',')
#dataset['Appliances']= dataset['Appliances'].astype('category')
# =============================================================================
# 
# #Convert datetime to ordinal value
# 
# dataset['date']= pd.to_datetime(dataset['date'])
# dataset['date'] = dataset['date'].map(dt.toordinal)
# #dataset['date'] = dataset['date'].map(dt.fromordinal)
# #dataset.Appliances.dtype
# =============================================================================

#Assigning dependent and independent variables
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


#Remove selected columns after feature importance
#Energy value greater than 50
# =============================================================================
# d1=dataset[0:0]
# i=0
# 
# for i in range(19735):
#     if dataset['Appliances'][i] < 50:
#         d1[i]=dataset[i]
# 
# dfil = dataset[(dataset.Appliances > 50)]
# print(dfil)
# 
# X1= X
# X1=pd.DataFrame(dataset)
# #X1=X1.drop("T9","Visibility","rv1","rv2", axis=0)
# X1.columns= dataset.columns
# X1.drop(X1.columns[[18,26,27]], axis=1, inplace=True)
# X = X1.iloc[:, :-1].values
# =============================================================================

#X.drop(X.columns[[18,26,27]], axis=1, inplace=True)

# =============================================================================
# """
# Y=Y.reshape(-1,1)
# Y = pd.DataFrame(Y)
# Y.columns = ['Appliances']
# Y['Appliances']= Y['Appliances'].astype('category')
# Y.Appliances.dtype
# """
# =============================================================================
#Replace missing values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:27])
X[:, 1:27] = imputer.transform(X[:, 1:27])

#Split the data between the Training Data and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.35, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm_X = MinMaxScaler()
X_train = mm_X.fit_transform(X_train)
X_test = mm_X.transform(X_test)

import csv
csvfile = "\\tholospg.itserv.scss.tcd.ie\Pgrad\rameshsu\Desktop\Suprith\ML\MLGit\tempValues.xlsx"
#RandomForest
#numArr = [210,220,230,240,250,260,270,280,290,300]
numArr = [410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600]
#numArr = [200,210,220]
for x in numArr:
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=x, random_state=0)  
    regressor.fit(X_train, Y_train)  
    y_pred = regressor.predict(X_test)
    print(regressor.feature_importances_)
    rms_rf = sqrt(mean_squared_error(Y_test, y_pred))
    rsqrd_rf = r2_score(Y_test, y_pred)
    mae_rf = mean_absolute_error(Y_test, y_pred)
    res=["n="+str(x),str(rms_rf),str(rsqrd_rf),'0',str(mae_rf)]
    
    #Assuming res is a flat list
    myFile = open("tempValues.xls", "a")  
    with myFile:  
        writer = csv.writer(myFile, lineterminator='\t')
        #writer.writerows(res)    
        for val in res:
           writer.writerow([val])   
        myFile.write("\n")
        #for val in res:
           # writer.writerow([val])    

#mape_rf = mean_absolute_percentage_error(Y_test, y_pred)
#score_rf = metrics.accuracy_score(Y_test, y_pred)

# GB ALGORITHM
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
regressor1 = GradientBoostingRegressor(learning_rate=0.5,n_estimators=400, loss='ls')
regressor1.fit(X_train, Y_train)    
y_pred1 = regressor1.predict(X_test)
rms1 = sqrt(mean_squared_error(Y_test, y_pred1))
rsqrd1=r2_score(Y_test, y_pred1)

# =============================================================================
# #SVR Algorithm
# from sklearn.svm import SVR
# regressor_svr=SVR(kernel='rbf',C=1000)
# regressor_svr.fit(X_train, Y_train)
# y_pred_svr = regressor_svr.predict(X_test)
# rms_svr = sqrt(mean_squared_error(Y_test, y_pred_svr))
# rsqrd_svr=r2_score(Y_test, y_pred_svr)
# mae_svr = mean_absolute_error(Y_test, y_pred_svr)
# =============================================================================
# =============================================================================
# #Multiple Linear Regression
# from sklearn.linear_model import LinearRegression
# regressor_lr = LinearRegression()
# regressor_lr.fit(X_train, Y_train)
# print(regressor_lr.score(X_train, Y_train))
# y_pred_lr = regressor_lr.predict(X_test)
# rms_lr = sqrt(mean_squared_error(Y_test, y_pred_lr))
# rsqrd_lr=r2_score(Y_test, y_pred_lr)
# mae_lr = mean_absolute_error(Y_test, y_pred_lr)
# =============================================================================
# =============================================================================
# # XGBoost
# import xgboost as xgb
# regressor4 = xgb.XGBRegressor()
# regressor4.fit(X_train, Y_train)
# y_pred4 = regressor4.predict(X_test)
# rms4 = sqrt(mean_squared_error(Y_test, y_pred4))
# rsqrd4=r2_score(Y_test, y_pred4)
# =============================================================================


#xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
