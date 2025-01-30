import numpy as np
import pandas as pd
from utility import *

class LogisticRegression:
    """
    A logistic regression model for binary classification.
    
    Attributes:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for training.
        weights (numpy.ndarray): Weights of the logistic regression model.
        bias (float): Bias term of the logistic regression model.
    """
    def __init__(self, learning_rate=0.001, num_iterations=1000, debug=False):
        """
        Initializes the logistic regression model with specified parameters.
        
        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.debug = debug
    
    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.
        
        Parameters:
        z (numpy.ndarray): Input array.
        
        Returns:
        numpy.ndarray: Sigmoid of the input.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Trains the logistic regression model on the given data.
        
        Parameters:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Target labels.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        if self.debug:
                print("Weights Before: ", self.weights)
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # if i % 1000 == 0:
            #     loss = self.compute_loss(y, y_predicted)
            #     print(f'Iteration {i}: Loss {loss}')
        if self.debug:
            print("Weights After: ", self.weights)
    def compute_loss(self, y, y_pred):
        """
        Computes the loss per layer.
        
        Parameters:
        y (numpy.ndarray): Actual y values of the dataset
        y_pred (numpy.ndarray): Predicted y values
        
        Returns:
        The calculated loss
        """
        num_samples = len(y)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Clip to prevent log(0), program was crashing during testing without this
        loss = -(1 / num_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def predict(self, X):
        """
        Predicts binary labels for the given input data.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Predicted binary labels.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

class LinearRegression:
    """
    A linear regression model for regression tasks.
    
    Attributes:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for training.
        weights (numpy.ndarray): Weights of the linear regression model.
        bias (float): Bias term of the linear regression model.
    """
    def __init__(self, learning_rate=0.001, num_iterations=1000, debug=False):
        """
        Initializes the linear regression model with specified parameters.
        
        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.debug = debug

    def fit(self, X, y):
        """
        Trains the linear regression model on the given data.
        
        Parameters:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Target values.
        """
        self.m, self.n = X.shape
        self.weights = np.zeros((self.n, 1))
        self.bias = 0
        self.X = X
        self.y = y.reshape(self.m, 1)

        for _ in range(self.num_iterations):
            self.update_weights()

    def update_weights(self):
        """
        Updates the internal weights of the model
        """
        if self.debug:
            print("Weights Before: ", self.weights)
        y_pred = self.predict(self.X)
        dW = -(2 * (self.X.T).dot(self.y - y_pred)) / self.m
        db = -2 * np.sum(self.y - y_pred) / self.m

        # Gradient clipping to prevent overflow, had issues with models crashing without this
        dW = np.clip(dW, -1, 1)
        db = np.clip(db, -1, 1)

        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * db
        if self.debug:
            print("Weights After: ", self.weights)
    def predict(self, X):
        """
        Predicts continuous values for the given input data.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Predicted continuous values.
        """
        return X.dot(self.weights) + self.bias
    
if __name__ == "__main__":
    # Example for Logistic Regression with Breast Cancer Dataset
    breast_cancer_path = '../Data/breastcancer/breast-cancer-wisconsin.data'
    breast_cancer_columns = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                             'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
                             'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
    breast_cancer = pd.read_csv(breast_cancer_path, header=None, names=breast_cancer_columns, na_values='?')
    breast_cancer.fillna(breast_cancer.median(), inplace=True)
    breast_cancer['Bare_Nuclei'] = pd.to_numeric(breast_cancer['Bare_Nuclei'], errors='coerce')
    breast_cancer.fillna(breast_cancer.median(), inplace=True)
    breast_cancer['Class'] = breast_cancer['Class'].apply(lambda x: 1 if x == 4 else 0)
    y = breast_cancer['Class'].values
    X = breast_cancer.drop(columns=['ID', 'Class']).values
    X_breast_cancer_normalized = min_max_normalization(X)

    X_train, X_test, y_train, y_test = train_test_split(X_breast_cancer_normalized, y, test_size=0.2, random_state=42)
    
    logistic_regression = LogisticRegression(learning_rate=0.0001, num_iterations=100000)
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy}")
    
    # Example for Linear Regression with Abalone Dataset
    abalone_path = '../Data/abalone/abalone.data'
    abalone_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    abalone = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    abalone['Sex'] = abalone['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    y = abalone['Rings'].values.reshape(-1, 1)
    X = abalone.drop(columns=['Rings']).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    linear_regression = LinearRegression(learning_rate=0.01, num_iterations=100000)
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression Mean Squared Error: {mse}")
    print("R-Squared:", r_squared(y_test, y_pred))
