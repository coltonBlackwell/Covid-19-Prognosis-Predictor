import numpy as np
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k):
        self.k = k
        self.training_features = None
        self.training_labels = None

    def fit(self, X, y):
        self.training_features = X
        self.training_labels = y

    def euclideanDistance(self, sample1, sample2):
        diff = sample1 - sample2
        squared_diff = diff ** 2
        sum_squared_diff = np.sum(squared_diff, axis=1)
        distance = np.sqrt(sum_squared_diff)
        return distance

    def nearestNeighbors(self, test_sample):
        distances = self.euclideanDistance(self.training_features, test_sample)
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:self.k]
        neighbors = [self.training_labels[i] for i in nearest_indices]
        return neighbors
    
    def predict(self, test_set):
        predictions=[]
        for test_sample in test_set:
            neighbors=self.nearestNeighbors(test_sample)
            labels=[sample for sample in neighbors]
            prediction=max(labels,key=labels.count)
            predictions.append(prediction)
        return predictions
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        return {'k': self.k}
    
    def set_params(self, **params):
        self.k = params['k']
        return self
