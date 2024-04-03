import pandas as pd
import numpy as np
from KNN import KNN
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import randint

# Below dataset for training KNN model

df = pd.read_csv("/home/colton/Documents/university/3rd Year/2nd Semester/CMPT 459/Assignments/CMPT459-Final-Group-Project/Steps-4-5/result/oversampled_processed_data.csv")
df = df.drop(columns=['outcome_group', 'sex', 'province', 'country', 'chronic_disease_binary'])

df = df[0:10000]

X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# --------------------------------------------------------------------- Fitting/training KNN model

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test) #avoid data leakage


model=KNN(3) #our model
model.fit(X_train,y_train)
   

predictions=model.predict(X_test)#our model's predictions

cm = confusion_matrix(y_test, predictions) #our model
print(cm)
print(accuracy_score(y_test, predictions)) 

print(classification_report(y_test, predictions))

# --------------------------------------------------------------------- Visualizing KNN model


# Define colormap for classes
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  # Adding 3D projection to the subplot

# Scatter plot with 3D settings
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap, edgecolors='k', s=50, alpha=0.8)

# Adding labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot with Colored Classes')

# Adding color bar
plt.colorbar(scatter, ax=ax, label='Class')

plt.tight_layout()
# plt.show()

# --------------------------------------------------------------------- Hyperparameter tuning

hyperParam_tuning = pd.read_csv("/home/colton/Documents/university/3rd Year/2nd Semester/CMPT 459/Assignments/CMPT459-Final-Group-Project/Steps-6/hyperparameter_tuning_data/hyperparameter_tuning_data.csv")
hyperParam_tuning = hyperParam_tuning.drop(columns=['sex', 'province', 'country', 'chronic_disease_binary', 'Combined_Key', 'outcome'])

print(np.array(hyperParam_tuning).shape)

# hyperParam_tuning = hyperParam_tuning[23519:33519]



X = hyperParam_tuning.iloc[:,:-1].values
y = hyperParam_tuning.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


error = []

for i in range(1, 10):
    print("Cycle complete")
    knn = KNN(i)
    knn.fit(X_train, y_train)
    predictions=knn.predict(X_test)#our model's predictions
    error.append(np.mean(predictions != y_test))


cm = confusion_matrix(y_test, predictions) #our model
print(cm)
print(accuracy_score(y_test, predictions)) 

plt.figure(figsize=(12,8))
plt.plot(range(1,10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title("Error rate for K value")
plt.xlabel('K Value')
plt.ylabel('Mean Error')
# plt.show()

# --------------------------------------------------------------------- GRID SEARCH CV (WORKS!!)

# Define the KNN classifier with a fixed k value of 3
knn = KNN(3)

# Define the parameter grid for grid search
param_grid = {
    'k': range(1, 5)  # Example parameter range for KNN
}

k_fold = KFold(n_splits=4, shuffle=True, random_state=0)

# Create and fit GridSearchCV object
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=k_fold, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score (Accuracy):", grid_search.best_score_)

# --------------------------------------------------------------------------------------------below is new


# Save results to a .txt file (replace 'results_knn.txt' with your desired file name)
with open('results_knn.txt', 'w') as file:
    file.write(str(grid_search.cv_results_))

# Evaluate the best model on test data (replace X_test, y_test with your actual test data)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

macro_f1_test = f1_score(y_test, y_pred, average='macro', labels=[1], zero_division=1)


macro_f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

print("DECEASED:", macro_f1_test)
print("Mean macro F1-score on test data:", macro_f1)
print("Mean overall accuracy on test data:", accuracy)


# -----------------------------------------------------------------



# --------------------------------------------------------------------- RANDOM SEARCH CV

# knn = KNN(3)

# param_dist = {
#     # 'k': [1,2,3,4,5,6,7,8,9,10]  # Example parameter range for KNN
#     'k': randint(1,10)  # Example parameter range for KNN

# }

# # Create and fit GridSearchCV object
# random_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, cv=5, n_iter=5, random_state=0)
# random_search.fit(X_train, y_train)

# # Print best parameters and best score
# print("Best Parameters:", random_search.best_params_)
# print("Best Score (Accuracy):", random_search.best_score_)

# -----------------------------------------------------------------

