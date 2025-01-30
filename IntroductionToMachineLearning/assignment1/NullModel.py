import numpy as np
import random

# Null model for classification
class ClassificationNullModel:
    def __init__(self):
        """
        Constructor for ClassificationNullModel.
        
        Initializes the class_counts dictionary to keep track of the count of each class label.
        Initializes the most_common_labels list to store the most frequent class labels.
        """
        self.class_counts = {}
        self.most_common_labels = []

    def fit(self, y):
        """
        Fit the model to the provided labels.
        
        Parameters:
        y (list): List of class labels.
        
        Iterates over the labels in y and updates the class_counts dictionary with the count of each label.
        Determines the most frequent class labels and stores them in most_common_labels.
        """
        for label in y:
            if label in self.class_counts:
                self.class_counts[label] += 1
            else:
                self.class_counts[label] = 1

        max_count = max(self.class_counts.values())
        self.most_common_labels = [cls for cls, count in self.class_counts.items() if count == max_count]

    def predict(self, X):
        """
        Predict the class labels for the provided input data.
        
        Parameters:
        X (list): List of input data points.
        
        Returns:
        list: List of predicted class labels.
        
        If there are multiple most frequent class labels, randomly selects one for each input data point, made it more interesting to me, gets a little boring when its all 1 type if theyre equal.
        Otherwise, assigns the most frequent class label to all input data points.
        """
        if len(self.most_common_labels) > 1:
            return [random.choice(self.most_common_labels) for _ in range(len(X))]
        else:
            return [self.most_common_labels[0] for _ in range(len(X))]

# Null model for regression
class RegressionNullModel:
    def fit(self, y):
        """
        Fit the model to the provided labels.
        
        Parameters:
        y (list): List or array of target values.
        
        Computes the mean of the target values and stores it.
        """
        self.mean_value = np.mean(y)

    def predict(self, X):
        """
        Predict the target values for the provided input data.
        
        Parameters:
        X (list): List or array of input data points.
        
        Returns:
        numpy.ndarray: Array of predicted target values.
        
        Assigns the mean of the training target values to all input data points.
        """
        return np.full((len(X),), self.mean_value)

if __name__ == "__main__":
    # Testing classification
    print("Testing ClassificationNullModel")

    # Simplistic test data to ensure functionality
    y_classification_train = ['M', 'B', 'B', 'B', 'M', 'B'] #If you change two of these B's to M's it will predict M for everything
    y_classification_test = ['M', 'B', 'B', 'M', 'B']

    model_classification = ClassificationNullModel()
    model_classification.fit(y_classification_train)
    predictions_classification = model_classification.predict(y_classification_test)
    print("Classification Predictions:", predictions_classification)

    # Testing regression
    print("Testing RegressionNullModel")

    # Simplistic test data to ensure functionality
    y_regression_train = [10, 20, 20, 30, 10, 30]
    y_regression_test = [15, 25, 35]

    model_regression = RegressionNullModel()
    model_regression.fit(y_regression_train)
    predictions_regression = model_regression.predict(y_regression_test)
    print("Regression Predictions:", predictions_regression)