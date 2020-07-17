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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

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

"""
---------------Training the Model---------------
"""
# Training the XGBoost classifier on the training data
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

"""
---------------Predicting Single Result---------------
"""
single_prediction = classifier.predict(np.array([X_test[0, :]]))
print("Single Prediction")
print(f"Single prediction is {single_prediction}")
print(f"Actual answer is {y_test[0]}", end="\n\n")

"""
---------------Predicting Test Set Result---------------
"""
y_pred = classifier.predict(X_test)
# Convert vectors from horizontal to vertical and display side by side to compare
print("Prediction, Actual")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

"""
--------------Making the Confusion Matrix and Evaluating the Model---------------
"""

# Gives confusion matrix C such that Cij is equal to the number of observations known to be in group i and predicted to be in group j.
cm = confusion_matrix(y_test, y_pred)
true_positive = cm[1][1]
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]

print("\nModel Evaluation")
print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
print(f"Precision is {true_positive / (true_positive + false_positive)}")
print(f"Recall is {true_positive / (true_positive + false_negative)}")
print(f"F1 score is {f1_score(y_test, y_pred, average='binary')}")
