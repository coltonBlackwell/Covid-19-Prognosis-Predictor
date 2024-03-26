from math import sqrt

# ---------------------------------------------------------VERSION 2

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
      
      neighbors.append(distances[i][0])
    return neighbors
  

  def predict(self,test_set):
    predictions=[]
    for test_sample in test_set:
      neighbors=self.nearest_neighbors(test_sample)
      labels=[sample for sample in neighbors]
      prediction=max(labels,key=labels.count)
      predictions.append(prediction)

    return predictions

