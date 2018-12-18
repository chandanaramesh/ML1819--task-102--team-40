import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score
from math import sqrt

dataset = pd.read_csv(r'energydata.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:30])
X[:, 1:30] = imputer.transform(X[:, 1:28])

#Split the data between the Training Data and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm_X = MinMaxScaler()
X_train = mm_X.fit_transform(X_train)
X_test = mm_X.transform(X_test)

#Correlation Matrix
dataset=dataset[0:30]
correlations = dataset.corr()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 12
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,30,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.grid(True)
plt.title('Feature Correlation')
labels =['date', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press', 'RH_out', 'Wind', 'Vis', 'Tdew', 'rv1', 'rv2', 'energy']
ax.set_xticklabels(labels,fontsize=6)
ax.set_yticklabels(labels,fontsize=6)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()

#SVR Algorithm
from sklearn.svm import SVR
regressor_svm=SVR(kernel='rbf')
regressor_svm.fit(X_train, Y_train)
y_pred_svm = regressor_svm.predict(X_test)
rms_svm = sqrt(mean_squared_error(Y_test, y_pred_svm))
rsqrd_svm=r2_score(Y_test, y_pred_svm)
mae_svm = mean_absolute_error(Y_test,y_pred_svm)

