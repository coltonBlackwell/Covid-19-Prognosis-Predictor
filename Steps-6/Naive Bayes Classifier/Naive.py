#gApril 6

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score

# Load the dataset 
data = pd.read_csv('cases_2021_train_processed_2 - cases_2021_train_processed_2.csv')

# # Split the dataset into features (X) and target variable (y)
# X = data.drop(columns=['Last_Update'])
# y = data['target_column']

# Split the data into training and validation sets (80:20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Naive Bayes classifier
nb_classifier = GaussianNB()

# Define the hyperparameters to tune
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

# Define K-fold cross-validation (K=5)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, 
                           scoring='f1_macro', cv=kfold)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_var_smoothing = grid_search.best_params_['var_smoothing']

# Train the Naive Bayes classifier with the best hyperparameters
best_nb_classifier = GaussianNB(var_smoothing=best_var_smoothing)
best_nb_classifier.fit(X_train, y_train)

# Predictions on the validation set
y_pred = best_nb_classifier.predict(X_val)

# Calculate evaluation metrics
macro_f1 = f1_score(y_val, y_pred, average='macro')
f1_deceased = f1_score(y_val, y_pred, average=None)[0]  # Assuming 'deceased' is the positive class
overall_accuracy = accuracy_score(y_val, y_pred)

# Report the results
print("Model Hyperparameters")
print("Mean macro F1-score across the validation sets:", macro_f1)
print("Mean F1-score on 'deceased' across the validation sets:", f1_deceased)
print("Mean overall accuracy across the validation sets:", overall_accuracy)
print("\n[Naive Bayes] var_smoothing=", best_var_smoothing)
