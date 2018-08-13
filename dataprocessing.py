# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv')
dataset2 = pd.read_csv('winequality-red1.csv')
dataset1 = pd.read_csv('winequality-white.csv')

dataframe = pd.read_csv("C:/Users/mudit/Desktop/Data Analytics/Wine details/winequality-red.csv",delimiter =";")
print(dataframe.head())
dataframe.to_csv('winequality-red1.csv',index = False)
'''dataframe = pd.read_csv("C:/Users/mudit/Desktop/Data Analytics/Wine details/winequality-white.csv",delimiter =";")
print(dataframe.head())
dataframe.to_csv('winequality-white1.csv',index = False)'''

X = dataset2.iloc[:, :-1].values
y = dataset2.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using the backward elimination process
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1599,1)).astype(float), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,2,3,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,2,3,4,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,2,3,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,2,5,6,7,9,10,11]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()