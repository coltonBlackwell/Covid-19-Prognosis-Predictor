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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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


# Define a function to calculate performance metrics for a given model and dataset
def evaluate_model(X_train, y_train, X_val, y_val, model):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
    return train_accuracy, val_accuracy, train_f1_macro, val_f1_macro



# --------------------------------------------------------------------- Visualizing Naive_Bayes model


# Define colormap for classes
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  # Adding 3D projection to the subplot

# Scatter plot with 3D settings
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=50, alpha=0.8)

# Adding labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Country')
ax.set_zlabel('Outcome Group')
ax.set_title('3D Scatter Plot with Colored Classes')

# Adding color bar
plt.colorbar(scatter, ax=ax, label='Class')

plt.tight_layout()
# plt.show()


# --------------------------------------------------------------------- Hyperparameter tuning

hyperParam_tuning = pd.read_csv("hyperparameter_tuning_data.csv")
hyperParam_tuning = hyperParam_tuning.drop(columns=['sex', 'province', 'country', 'chronic_disease_binary', 'Combined_Key', 'outcome'])

print(np.array(hyperParam_tuning).shape)

# hyperParam_tuning = hyperParam_tuning[33519:43519]


X = hyperParam_tuning.iloc[:,:-1].values
y = hyperParam_tuning.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


error = []

for i in range(1, 10):
    naive = GaussianNB()
    naive.fit(X_train, y_train)
    predictions=naive.predict(X_test)#our model's predictions
    error.append(np.mean(predictions != y_test))


cm = confusion_matrix(y_test, predictions) #our model
print("confusion matrix: ", cm)
print("accuracy score: ",accuracy_score(y_test, predictions)) 

# plt.figure(figsize=(12,8))
# plt.plot(range(1,10), error, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title("Error rate for K value")
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# # plt.show()



# --------------------------------------------------------------------- grid search

# Define the parameter grid for grid search
param_grid = {
    'k': range(3,7)  # Example parameter range for KNN
}

k_fold = KFold(n_splits=4, shuffle=True, random_state=0)

# Create and fit GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=k_fold, scoring='accuracy')



# --------------------------------------------------------------------- Kfold 


hyperParam_tuning = pd.read_csv("hyperparameter_tuning_data.csv")
hyperParam_tuning = hyperParam_tuning.drop(columns=['sex', 'province', 'country', 'chronic_disease_binary', 'Combined_Key', 'outcome'])

# Define K-fold cross-validation (K=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

hyperParam_tuning = hyperParam_tuning[33519:43519]


X = hyperParam_tuning.iloc[:,:-1].values
y = hyperParam_tuning.iloc[:, -1].values


# Initialize lists to store accuracy scores for each fold
accuracy_scores = []
# Initialize lists to store F1-scores for each fold
f1_scores = []
# Initialize lists to store F1-scores for 'deceased' class for each fold
f1_deceased_scores = []
# Initialize lists to store performance metrics for each fold
train_accuracy_scores = []
val_accuracy_scores = []
train_f1_macro_scores = []
val_f1_macro_scores = []

# Perform K-fold cross-validation
for train_index, val_index in kfold.split(X_train):

    x = hyperParam_tuning.iloc[:,:-1].values
    y = hyperParam_tuning.iloc[:, -1].values

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Fit the classifier on the training fold
    model.fit(x, y)
    
    # Predict on the validation fold
    y_pred_fold = model.predict(x)
    
    # Calculate accuracy on the validation fold
    accuracy_fold = accuracy_score(y, y_pred_fold)
    accuracy_scores.append(accuracy_fold)

    # Calculate F1-score on the validation fold
    f1_fold = f1_score(y, y_pred_fold, average='macro')
    f1_scores.append(f1_fold)

    # Calculate F1-score for 'deceased' class on the validation fold
    f1_deceased_fold = f1_score(y, y_pred_fold, average=None)[0]  # Assuming 'deceased' is the first class
    f1_deceased_scores.append(f1_deceased_fold)

    # Evaluate the model on the current fold
    train_accuracy, val_accuracy, train_f1_macro, val_f1_macro = evaluate_model(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, model)


# Calculate mean accuracy across all folds
mean_accuracy = np.mean(accuracy_scores)
# Calculate mean macro F1-score across all folds
mean_macro_f1 = np.mean(f1_scores)
# Calculate mean F1-score for 'deceased' class across all folds
mean_f1_deceased = np.mean(f1_deceased_scores)


# Calculate mean performance metrics across all folds
mean_train_accuracy = np.mean(train_accuracy_scores) #all nan 
mean_val_accuracy = np.mean(val_accuracy_scores)
mean_train_f1_macro = np.mean(train_f1_macro_scores)
mean_val_f1_macro = np.mean(val_f1_macro_scores)

# Report mean performance metrics
print("Mean Training Accuracy:", mean_train_accuracy)
print("Mean Validation Accuracy:", mean_val_accuracy)
print("Mean Training F1-score (macro):", mean_train_f1_macro)
print("Mean Validation F1-score (macro):", mean_val_f1_macro)




print("Mean F1-score on 'deceased' across 5 folds:", mean_f1_deceased)
print("Mean macro F1-score across 5 folds:", mean_macro_f1)
print("Mean accuracy across 5 folds:", mean_accuracy) 
#Mean accuracy across 5 folds: 0.8359
#Mean accuracy across 9 folds: 0.8359 //why same value