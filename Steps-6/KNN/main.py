import pandas as pd
from KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Below dataset for training KNN model

df = pd.read_csv("/home/colton/Documents/university/3rd Year/2nd Semester/CMPT 459/Assignments/CMPT459-Final-Group-Project/Steps-4-5/result/oversampled_processed_data.csv")
df = df.drop(columns=['outcome_group', 'sex', 'province', 'country', 'chronic_disease_binary'])

# df = df[0:10000]

X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# --------------------------------------------------------------------- Fitting/training KNN model

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) #avoid data leakage


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
plt.show()

# --------------------------------------------------------------------- Hyperparameter tuning

