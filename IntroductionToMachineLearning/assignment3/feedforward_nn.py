import numpy as np
import pandas as pd
from utility import *

class FeedforwardNeuralNetwork:
    """
    A feedforward neural network with two hidden layers for classification or regression.
    
    Attributes:
        input_size (int): Number of input features.
        hidden_sizes (list of int): Number of nodes in each hidden layer.
        output_size (int): Number of output nodes.
        activation (str): Activation function for hidden layers.
        output_activation (str): Activation function for output layer.
        W1 (numpy.ndarray): Weights between input layer and first hidden layer.
        b1 (numpy.ndarray): Biases for first hidden layer.
        W2 (numpy.ndarray): Weights between first hidden layer and second hidden layer.
        b2 (numpy.ndarray): Biases for second hidden layer.
        W3 (numpy.ndarray): Weights between second hidden layer and output layer.
        b3 (numpy.ndarray): Biases for output layer.
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh', output_activation='linear', loss='mse'):
        """
        Initializes the feedforward neural network with given sizes and activation functions.
        
        Parameters:
        input_size (int): Number of input features.
        hidden_sizes (list of int): Number of nodes in each hidden layer.
        output_size (int): Number of output nodes.
        activation (str): Activation function for hidden layers.
        output_activation (str): Activation function for output layer.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def activation_function(self, x):
        """
        Computes the activation function.
        
        Parameters:
        z (numpy.ndarray): Input array.
        function (str): Activation function to use ('tanh' or 'sigmoid').
        
        Returns:
        numpy.ndarray: Activated output.
        """
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'logistic':
            return 1 / (1 + np.exp(-x))
        return x

    def activation_derivative(self, x):
        """
        Computes the derivative of the activation function.
        
        Parameters:
        z (numpy.ndarray): Input array.
        function (str): Activation function to use ('tanh' or 'sigmoid').
        
        Returns:
        numpy.ndarray: Derivative of the activated output.
        """
        if self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'logistic':
            return self.activation_function(x) * (1 - self.activation_function(x))
        return np.ones_like(x)

    def output_activation_function(self, x):
        """
        Computes the activation function

        Parameters:
        x (numpy.ndarray): input array

        Returns:
        If softmax returns the calculated version, if not returns X.
        """
        if self.output_activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return x

    def loss_function(self, y_true, y_pred):
        """
        Computes the Loss Function.

        Parameters:
        y_true (numpy.ndarray): Actual y values of the dataset
        y_pred (numpy.ndarray): Predicted y values

        Returns:
        mse or cross entropy depending on what is selected as the self.loss
        """
        if self.loss == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss == 'cross_entropy':
            return -np.mean(y_true * np.log(y_pred + 1e-8))
        return np.mean((y_true - y_pred) ** 2)

    def loss_derivative(self, y_true, y_pred):
        """
        Computes the derivative of the loss functions

        Parameters:
        y_true (numpy.ndarray): Actual y values of the dataset
        y_pred (numpy.ndarray): Predicted y values

        returns y_pred - y_true
        """
        return y_pred - y_true

    def forward(self, X):
        """
        Performs the forward pass of the network.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Network output.
        """
        self.layer_inputs = []
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            input_to_layer = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(input_to_layer)

            if i == len(self.weights) - 1:
                output = self.output_activation_function(input_to_layer)
            else:
                output = self.activation_function(input_to_layer)

            self.layer_outputs.append(output)
        return self.layer_outputs[-1]

    def backward(self, X, y, learning_rate):
        """
        Performs the backward pass and updates weights and biases.
        
        Parameters:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): Target labels.
        learning_rate (float): Learning rate for weight updates.
        """
        y_pred = self.forward(X)
        loss_derivative = self.loss_derivative(y, y_pred)
        deltas = [loss_derivative]

        # Calculate deltas for each layer
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.activation_derivative(self.layer_inputs[i])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            grad_w = np.dot(self.layer_outputs[i].T, deltas[i])
            grad_b = np.mean(deltas[i], axis=0)
            grad_w = np.clip(grad_w, -1, 1)
            grad_b = np.clip(grad_b, -1, 1)
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b

    def fit(self, X, y, epochs=1000, learning_rate=0.001):
        """
        Trains the feedforward neural network on the given data.
        
        Parameters:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Target labels.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        """
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                y_pred = self.forward(X)
                loss = self.loss_function(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Predicts the output using the trained network.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Network output.
        """
        return self.forward(X)
    
if __name__ == "__main__":
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

    X_train, X_test, y_train, y_test = train_test_split(X_breast_cancer_normalized, y, test_size=0.2)
    y_train_one_hot = one_hot_encode(y_train, num_classes=2)
    nn_classification = FeedforwardNeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[10, 10], output_size=2, activation='tanh', output_activation='softmax', loss='cross_entropy')
    nn_classification.fit(X_train, y_train_one_hot, epochs=1000, learning_rate=0.001)
    predictions_classification = nn_classification.predict(X_test)
    y_pred_classification = np.argmax(predictions_classification, axis=1)
    
    # Classification Metrics
    print("Classification Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred_classification))
    print("Precision:", precision_score(y_test, y_pred_classification))
    print("Recall:", recall_score(y_test, y_pred_classification))
    print("F1 Score:", f1_score(y_test, y_pred_classification))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classification))

    abalone_path = '../Data/abalone/abalone.data'
    abalone_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    abalone = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    abalone['Sex'] = abalone['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    y = abalone['Rings'].values.reshape(-1, 1)
    X = abalone.drop(columns=['Rings']).values
    X_abalone_normalized = min_max_normalization(X)
    y_normalized = min_max_normalization(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_abalone_normalized, y_normalized, test_size=0.2)
    nn_regression = FeedforwardNeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[10, 10], output_size=1, activation='tanh', output_activation='linear', loss='mse')
    nn_regression.fit(X_train, y_train, epochs=1000, learning_rate=0.001)
    predictions_regression = nn_regression.predict(X_test)    
    # Regression Metrics
    print("\nRegression Metrics:")
    print("Mean Squared Error:", mean_squared_error(y_test, predictions_regression))
    print("R-Squared:", r_squared(y_test, predictions_regression))
