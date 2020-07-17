"""
This is a program to train a classifier using the XGBoost ML algorithm to detect the presence
of parkinsons in individuals
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
---------------Data Pre-processing---------------
"""

dataset = pd.read_csv("Data/parkinsons.data")
print(dataset.head(), end="\n\n") # view first 5 rows of data to see its structure.

# X is the matrix of features and y is the vector of labels
# X doesn't contain the label or the patient names which is the first column
X = dataset.loc[:, dataset.columns != 'status'].values[:, 1:]
y = dataset.loc[:, 'status'].values

# Find out how many of each label type there is
print(f"There are {y[y == 0].shape[0]} 0 labels")
print(f"There are {y[y == 1].shape[0]} 1 labels", end = "\n\n")

# Split the data into training sets and test sets. 80% of data used for training
# Splitting before scaling as to not include test set data and provide future information
# to the model. Only test set used to scale.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Apply feature scaling to training set and test set
scaler = MinMaxScaler((-1, 1)) # Scales features between -1 and 1
# X_test uses same scaler values calculated from the training set.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
