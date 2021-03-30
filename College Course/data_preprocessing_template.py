# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:38:34 2018

@author: patil
"""
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('Data.csv')

# Creating the matrix of features. In the imported dataset Country, Age, Salary are independent variables
X = dataset.iloc[:, :-1].values

# Creating the dependent variable
y = dataset.iloc[:, 3].values
...
# Taking care of missing data
from sklearn.preprocessing import Imputer

# Creating the object of Imputer class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# fit imputer object to data X (Matrix of feature X)
imputer = imputer.fit(X[:, 1:3]) 

# Replace the missing data of column by mean
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Creating dummy variables using OneHotEncoder class
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

# Creating the object of LabelEncoder class
labelencoder_X = LabelEncoder()

# fit labelencoder_X object to first coulmn Country of matrix X
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Creating dummy variables
# Creating the object of OneHotEncoder class
onehotencoder = OneHotEncoder(categorical_features = [0])

# fit onehotencoder object to first column - Country of matrix X
X = onehotencoder.fit_transform(X).toarray()

# Creating the object of LabelEncoder class
labelencoder_y = LabelEncoder()

# fit labelencoder_y object to last coulmn Purchased, we will get encoded vector
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split

# Choosing 20% data as test data, so we will have 80% data in training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# Creating the object of StandardScaler
sc_X = StandardScaler()

# fit and transform training set
X_train = sc_X.fit_transform(X_train)

# transform test set
X_test = sc_X.transform(X_test)