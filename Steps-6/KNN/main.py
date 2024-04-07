import pandas as pd
import numpy as np
from KNN import KNN
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from mpl_toolkits.mplot3d import Axes3D  # Importing Axes3D from mpl_toolkits.mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import randint
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA



# Below dataset for training KNN model

df = pd.read_csv("../../Steps-4-5/result/oversampled_processed_data.csv")
df = df.drop(columns=['outcome_group', 'sex', 'province', 'country', 'chronic_disease_binary'])

df = df[0:10000]

X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# --------------------------------------------------------------------- Fitting/training KNN model

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


model=KNN(1) #our model
model.fit(X_train,y_train)
   

predictions=model.predict(X_test)#our model's predictions


# --------------------------------------------------------------------- Visualizing KNN model


# Define colormap for classes
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  # Adding 3D projection to the subplot

# Scatter plot with 3D settings
scatter = ax.scatter(X[:, 0], X[:, 11], X[:, 12], c=y, cmap=cmap, edgecolors='k', s=50, alpha=0.8)

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

hyperParam_tuning = pd.read_csv("../data/hyperparameter_tuning_data/hyperparameter_tuning_data.csv")
hyperParam_tuning = hyperParam_tuning.drop(columns=['sex', 'province', 'country', 'chronic_disease_binary', 'Combined_Key', 'outcome'])

label_support = hyperParam_tuning['outcome_group_code'].value_counts()
print(label_support)


print(np.array(hyperParam_tuning).shape)

# hyperParam_tuning = hyperParam_tuning[0:43519]


X = hyperParam_tuning.iloc[:,:-1].values
y = hyperParam_tuning.iloc[:, -1].values


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# error = []

# for i in range(1, 10):
#     print("Cycle complete")
#     knn = KNN(i)
#     knn.fit(X_train, y_train)
#     predictions=knn.predict(X_test)#our model's predictions
#     error.append(np.mean(predictions != y_test))


# cm = confusion_matrix(y_test, predictions) #our model
# print(cm)
# print(accuracy_score(y_test, predictions)) 

# plt.figure(figsize=(12,8))
# plt.plot(range(1,10), error, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title("Error rate for K value")
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# plt.show()

# --------------------------------------------------------------------- GRID SEARCH CV (WORKS!!)

# target_class_label = 1

# x_filtered = X[y == target_class_label]
# y_filtered = y[y == target_class_label]



# # Define the KNN classifier with a fixed k value of 3
# knn = KNN(3)

# # Define the parameter grid for grid search
# param_grid = {
#     'k': range(3, 7)  # Example parameter range for KNN
# }

# k_fold = KFold(n_splits=4, shuffle=True, random_state=0)

# # Create and fit GridSearchCV object
# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=k_fold, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # Print best parameters and best score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score (Accuracy):", grid_search.best_score_)

# # --------------------------------------------------------------------------------------------below is new


# # Save results to a .txt file (replace 'results_knn.txt' with your desired file name)
# with open('results_knn_NEW.txt', 'w') as file:
#     file.write(str(grid_search.cv_results_))

# # Evaluate the best model on test data (replace X_test, y_test with your actual test data)
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)


# macro_f1_test = f1_score(y_test, y_pred, average='macro', labels=[1], zero_division=1)


# macro_f1 = f1_score(y_test, y_pred, average='macro')
# accuracy = accuracy_score(y_test, y_pred)

# # print("DECEASED:", macro_f1_test)
# print("Mean macro F1-score on test data:", macro_f1)
# print("Mean overall accuracy on test data:", accuracy)

# --------------------------------------------------------------------------------------------below is new


knn = KNN(3)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the parameter grid for grid search
param_grid = {
    'k': range(3,7)  # Example parameter range for KNN
}

k_fold = KFold(n_splits=4, shuffle=True, random_state=0)

# Create and fit GridSearchCV object
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=k_fold, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)



# Calculate precision, recall, F1-score, and support for each class
report = classification_report(y_test, y_pred, labels=[1,2,4], target_names=['Deceased', 'Hospitalized', 'Recovered'], output_dict=True)

# Calculate macro and weighted average scores
macro_avg_f1 = report['macro avg']['f1-score']
weighted_avg_f1 = report['weighted avg']['f1-score']

# Print classification report and average scores
print(classification_report(y_test, y_pred, labels=[1,2,4], target_names=['Deceased', 'Hospitalized', 'Recovered']))
print("Mean macro F1-score on test data:", macro_avg_f1)
print("Mean weighted F1-score on test data:", weighted_avg_f1)