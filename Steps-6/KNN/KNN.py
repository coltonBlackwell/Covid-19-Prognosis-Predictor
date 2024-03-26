from math import sqrt
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
        print(self.k)

    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = y_train

    def calculate_euclidean(self, sample1, sample2):
        distance = np.sqrt(np.sum((sample1 - sample2)**2, axis=1))  # Calculate Euclidean distance for each sample
        return distance

    def nearest_neighbors(self, test_sample):
        distances = self.calculate_euclidean(self.x_train, test_sample)
        sorted_indices = np.argsort(distances)  # Get indices of sorted distances
        neighbors = self.y_train[sorted_indices[:self.k]]  # Get labels of k nearest neighbors
        return neighbors

    def predict(self, test_set):
        predictions = []
        for test_sample in test_set:
            neighbors = self.nearest_neighbors(test_sample)
            prediction = np.argmax(np.bincount(neighbors))  # Choose the most frequent label
            predictions.append(prediction)
        return predictions