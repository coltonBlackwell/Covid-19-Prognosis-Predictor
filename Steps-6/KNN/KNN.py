import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
        print(self.k)

    def fit(self, training_features, training_labels):
        self.training_features = training_features
        self.training_labels = training_labels

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
        distances = np.array([self.euclideanDistance(self.training_features, test_sample) for test_sample in test_set])
        nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
        neighbors = self.training_labels[nearest_indices]
        predictions = np.argmax(np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(self.training_labels))), axis=1, arr=neighbors), axis=1)
        return predictions