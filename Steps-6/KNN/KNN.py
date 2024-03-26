from math import sqrt
from collections import Counter

# class KNN:
#     def __init__(self, k):
#         self.k = k

#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train

#     def calculate_euclidean(self,sample1,sample2):
#         distance=0.0
#         for i in range(50, len(sample1)):
#             print(sample1[i])
#             print(sample1[i+1])
#             print(sample1[i+2])
#             print(sample1[i+3])
#             print(sample1[i+4])
#             print(sample1[i+5])
#             print(sample1[i+6])
        
#             distance+=(sample1[i]-sample2[i])**2 #Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)

#         print("THIS IS THE DISTANCE", distance)
#         return sqrt(distance)

#     def nearest_neighbors(self, test_sample):
#         distances = [(self.calculate_euclidean(sample, test_sample), label) for sample, label in zip(self.X_train, self.y_train)]
#         distances.sort(key=lambda x: x[0])  # Sort by distance
#         neighbors = [label for _, label in distances[:self.k]]
#         return neighbors

#     def predict(self, test_set):
#         predictions = []
#         for test_sample in test_set:
#             neighbors = self.nearest_neighbors(test_sample)
#             prediction = Counter(neighbors).most_common(1)[0][0]  # Majority vote
#             predictions.append(prediction)
#         return predictions


# ---------------------------------------------------------VERSION 2

from math import sqrt
class KNN():
  def __init__(self,k):
    self.k=k
    print(self.k)


  def fit(self,X_train,y_train):

    self.x_train=X_train
    self.y_train=y_train


  def calculate_euclidean(self,sample1,sample2):
    distance=0.0


    for i in range(len(sample1)):
      
      distance+=(sample1[i]-sample2[i])**2 #Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)

    return sqrt(distance)
  

  def nearest_neighbors(self,test_sample):
    distances=[]#calculate distances from a test sample to every sample in a training set
    for i in range(len(self.x_train)):
      
        distances.append((self.y_train[i],self.calculate_euclidean(self.x_train[i],test_sample)))
        distances.sort(key=lambda x:x[1])#sort in ascending order, based on a distance value
        neighbors=[]

    for i in range(self.k): #get first k samples
      
      neighbors.append(distances[i][1])

    return neighbors
  

  def predict(self,test_set):
    predictions=[]
    for test_sample in test_set:
      
      neighbors=self.nearest_neighbors(test_sample)
      labels=[sample for sample in neighbors]
      prediction=max(labels,key=labels.count)
      predictions.append(prediction)

    return predictions
  
# -----------------------------------------------------------------VERSION 1

# import numpy as np
# from collections import Counter

# def euclidean_distance(x1, x2):

#     print("X1 WOWOW:", x1)
#     print("X1 WOWOW:", x2)

#     distance = np.sqrt(np.sum((x1-x2)**2))
#     return distance

# class KNN:
#     def __init__(self, k=3):
#         self.k = k

    
#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y

#     def predict(self, X):
#         predictions = [self._predict(x) for x in X]
#         return predictions

#     def _predict(self, x):

#         #computing distance
#         distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

#         # Getting the closest K
#         k_indices = np.argsort(distances)[:self.k]
#         k_nearest_labels = [self.y_train[i] for i in k_indices]

#         #majority Vote
#         most_common = Counter(k_nearest_labels).most_common()
#         return most_common