#start over


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# Importing the dataset
dataset = pd.read_csv('oversampled_processed_data.csv')

#train the dataset
dataset = dataset.drop(columns=['outcome_group', 'sex', 'province', 'country', 'chronic_disease_binary'])

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
print(X_train)



# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict([X_test[6]])

print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])




y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)


# input_string = X_train
# # to_convert = re.findall(r'\d+\.\d+', input_string)

# # Type => X List
# # print(to_convert) # ['3.1417']

# # Type => ? Float
# converted = float(input_string[0])

# print(converted) # 3.1417