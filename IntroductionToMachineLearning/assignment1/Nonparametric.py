import numpy as np
from collections import Counter

#Distance Calculation
def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two points.
    
    Parameters:
    x1 (numpy.ndarray): First point.
    x2 (numpy.ndarray): Second point.
    
    Returns:
    float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """
    Calculates the Manhattan distance between two points.
    
    Parameters:
    x1 (numpy.ndarray): First point.
    x2 (numpy.ndarray): Second point.
    
    Returns:
    float: Manhattan distance between x1 and x2.
    """
    return np.sum(np.abs(x1 - x2))

# Gaussian kernel function
def gaussian_kernel(distance, bandwidth):
    """
    Computes the Gaussian kernel function.
    
    Parameters:
    distance (float): Distance value.
    bandwidth (float): Bandwidth parameter for the Gaussian kernel.
    
    Returns:
    float: Computed Gaussian kernel value.
    """
    return np.exp(-0.5 * ((distance / bandwidth) ** 2))

# Distance matrix calculation with vectorization
def calculate_distances(X1, X2, distance_metric='euclidean'):
    """
    Calculates the distances between the input data and the training data.
    
    Parameters:
    X1 (numpy.ndarray): Input feature matrix.
    X2 (numpy.ndarray): Training feature matrix.
    distance_metric (str): Type of distance metric ('euclidean' or 'manhattan').
    
    Returns:
    numpy.ndarray: Distance matrix.
    """
    num_test = X1.shape[0]
    num_train = X2.shape[0]
    distances = np.zeros((num_test, num_train))

    if distance_metric == 'euclidean':
        for i in range(num_test):
            distances[i, :] = np.sqrt(np.sum((X2 - X1[i]) ** 2, axis=1))
    elif distance_metric == 'manhattan':
        for i in range(num_test):
            distances[i, :] = np.sum(np.abs(X2 - X1[i]), axis=1)

    return distances

# k-Nearest Neighbors algorithm
class KNN:
    def __init__(self, k=3, mode='classification', gamma=0.1, distance_metric='euclidean'):
        """
        Initializes the KNN model.
        
        Parameters:
        k (int): Number of nearest neighbors to consider.
        mode (str): Mode of the model ('classification' or 'regression').
        gamma (float): Bandwidth parameter for the Gaussian kernel.
        distance_metric (str): Distance metric to use ('euclidean' or 'manhattan').
        """
        self.k = k
        self.mode = mode
        self.gamma = gamma
        self.distance_metric = distance_metric

    def fit(self, X, y):
        """
        Fits the KNN model to the training data.
        
        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Training labels array.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the labels for the given input data.
        
        Parameters:
        X (numpy.ndarray): Input feature matrix.
        
        Returns:
        numpy.ndarray: Predicted labels.
        """
        distances = calculate_distances(X, self.X_train, self.distance_metric)
        return self._predict(distances)

    def _predict(self, distances):
        """
        Predicts the labels based on the distance matrix.
        
        Parameters:
        distances (numpy.ndarray): Distance matrix.
        
        Returns:
        numpy.ndarray: Predicted labels.
        """
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            k_nearest_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            if self.mode == 'classification':
                y_pred[i] = np.argmax(np.bincount(k_nearest_labels.astype(int)))
            elif self.mode == 'regression':
                weights = gaussian_kernel(distances[i][k_nearest_indices], self.gamma)
                y_pred[i] = np.dot(weights, k_nearest_labels) / np.sum(weights)
        
        return y_pred

# Edited KNN Algorithm
class EditedKNN:
    def __init__(self, k=3, mode='classification', epsilon=0.1, max_iterations=100, gamma=0.1, distance_metric='euclidean'):
        """
        Initializes the EditedKNN model.
        
        Parameters:
        k (int): Number of nearest neighbors to consider.
        mode (str): Mode of the model ('classification' or 'regression').
        epsilon (float): Threshold for editing in regression mode.
        max_iterations (int): Maximum number of iterations for the editing process.
        gamma (float): Bandwidth parameter for the Gaussian kernel.
        distance_metric (str): Distance metric to use ('euclidean' or 'manhattan').
        """
        self.k = k
        self.mode = mode
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.distance_metric = distance_metric
        self.edited = False

    def fit(self, X, y, validation_set=None):
        """
        Fits the EditedKNN model to the training data.
        
        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Training labels array.
        validation_set (tuple): Optional validation set (X_val, y_val) for performance evaluation.
        """
        self.X_train = X
        self.y_train = y
        if validation_set:
            self.X_val, self.y_val = validation_set
        else:
            self.X_val, self.y_val = None, None
        self._edit()
        self.edited = True

    def edit_training_set(self, X, y, validation_set=None):
        """
        Edits the training set based on performance evaluation.

        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Training labels array.
        validation_set (tuple): Optional validation set (X_val, y_val) for performance evaluation.

        Returns:
        tuple: Edited training feature matrix and labels array.
        """
        self.X_train = X
        self.y_train = y
        if validation_set:
            self.X_val, self.y_val = validation_set
        else:
            self.X_val, self.y_val = None, None
        self._edit()
        self.edited = True  
        return self.X_train, self.y_train

    def _edit(self):
        """
        Internal method to edit the training set based on nearest neighbors' performance.
        """
        edited_X = np.copy(self.X_train)
        edited_y = np.copy(self.y_train)
        initial_performance = self._evaluate(self.X_val, self.y_val)
        performance_degraded = False
        iterations = 0

        while not performance_degraded and iterations < self.max_iterations:
            if len(edited_X) == 0 or len(edited_y) == 0:
                print("Empty training set. Stopping editing process.")
                break
            distances = calculate_distances(edited_X, edited_X, self.distance_metric)
            np.fill_diagonal(distances, np.inf)
            nearest_indices = np.argmin(distances, axis=1)

            if self.mode == 'classification':
                predictions = np.array([np.bincount(edited_y[nearest_indices[i:i+1]].astype(int)).argmax() for i in range(len(edited_y))])
                deletions = (predictions != edited_y)
            elif self.mode == 'regression':
                predictions = np.array([np.mean(edited_y[nearest_indices[i:i+1]]) for i in range(len(edited_y))])
                deletions = (np.abs(predictions - edited_y) > self.epsilon)

            if np.any(deletions):
                edited_X = edited_X[~deletions]
                edited_y = edited_y[~deletions]
            else:
                break

            current_performance = self._evaluate(self.X_val, self.y_val)
            if (self.mode == 'classification' and current_performance < initial_performance) or (self.mode == 'regression' and current_performance > initial_performance):
                performance_degraded = True
            iterations += 1

        if iterations == self.max_iterations:
            print("Reached maximum iterations without performance degradation.")
        else:
            print("Performance degraded. Stopping editing process.")

        self.X_train = edited_X
        self.y_train = edited_y

    def _evaluate(self, X, y):
        """
        Evaluates the model performance.
        
        Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Labels array.
        
        Returns:
        float: Performance metric (accuracy for classification, MSE for regression).
        """
        if X is None or y is None:
            return float('inf') if self.mode == 'regression' else 0
        predictions = self.predict(X)
        if self.mode == 'classification':
            accuracy = np.sum(predictions == y) / len(y)
            return accuracy
        elif self.mode == 'regression':
            mse = np.mean((predictions - y) ** 2)
            return mse

    def predict(self, X):
        """
        Predicts the labels for the given input data.
        
        Parameters:
        X (numpy.ndarray): Input feature matrix.
        
        Returns:
        numpy.ndarray: Predicted labels.
        """
        distances = calculate_distances(X, self.X_train, self.distance_metric)
        return self._predict(distances)
    
    def _predict(self, distances):
        """
        Predicts the labels based on the distance matrix.
        
        Parameters:
        distances (numpy.ndarray): Distance matrix.
        
        Returns:
        numpy.ndarray: Predicted labels.
        """
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            k_nearest_indices = np.argsort(distances[i])[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]

            if len(k_nearest_labels.astype(int)) == 0:
                y_pred[i] = np.nan
            elif self.mode == 'classification':
                y_pred[i] = np.argmax(np.bincount(k_nearest_labels.astype(int)))
            elif self.mode == 'regression':
                weights = gaussian_kernel(distances[i][k_nearest_indices], self.gamma)
                y_pred[i] = np.dot(weights, k_nearest_labels) / np.sum(weights)
        
        return y_pred