# import pandas as pd
# from KNN import KNN
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


# hyperParam_tuning = pd.read_csv("/home/colton/Documents/university/3rd Year/2nd Semester/CMPT 459/Assignments/CMPT459-Final-Group-Project/Steps-6/hyperparameter_tuning_data/hyperparameter_tuning_data.csv")
# hyperParam_tuning = hyperParam_tuning.drop(columns=['sex', 'province', 'country', 'chronic_disease_binary', 'Combined_Key'])

# X = hyperParam_tuning.iloc[:,:-1].values
# y = hyperParam_tuning.iloc[:, -1].values


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# error = []

# for i