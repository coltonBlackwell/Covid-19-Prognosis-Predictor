import pandas as pd
from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("/home/colton/Documents/university/3rd Year/2nd Semester/CMPT 459/Assignments/CMPT459-Final-Group-Project/Steps-4-5/result/oversampled_processed_data.csv")

df = df.drop(columns=['outcome_group', 'sex', 'province', 'country', 'chronic_disease_binary'])

X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train)

print(y_train)
# y = df['outcome_group_code']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test) #avoid data leakage


# # print(y_train)

# model = KNN(5)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# # print(predictions)