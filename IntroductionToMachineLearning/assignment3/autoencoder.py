import pandas as pd
import numpy as np
from utility import *

class Autoencoder:
    """
    A simple autoencoder for dimensionality reduction and feature extraction.
    
    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden nodes.
        W1 (numpy.ndarray): Weights between input layer and hidden layer.
        b1 (numpy.ndarray): Biases for hidden layer.
        W2 (numpy.ndarray): Weights between hidden layer and output layer.
        b2 (numpy.ndarray): Biases for output layer.
    """
    def __init__(self, input_size, hidden_size, debug=False):
        """
        Initializes the autoencoder with given input and hidden layer sizes.
        
        Parameters:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden nodes.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size)
        self.b2 = np.zeros((1, input_size))
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

    def sigmoid_derivative(self, z):
        """
        Computes the derivative of the sigmoid function.
        
        Parameters:
        z (numpy.ndarray): Input array.
        
        Returns:
        numpy.ndarray: Derivative of the sigmoid function.
        """
        return z * (1 - z)

    def forward(self, X):
        """
        Performs the forward pass of the autoencoder.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Reconstructed output.
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        if self.debug:
            print("Encode: ", self.A1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        if self.debug:
            print("Decode: ", self.A2)
        return self.A2

    def backward(self, X, output, learning_rate):
        """
        Performs the backward pass and updates weights and biases.
        
        Parameters:
        X (numpy.ndarray): Input data.
        output (numpy.ndarray): Reconstructed output.
        learning_rate (float): Learning rate for weight updates.
        """
        error = X - output
        dZ2 = error * self.sigmoid_derivative(output)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 += learning_rate * dW1
        self.b1 += learning_rate * db1
        self.W2 += learning_rate * dW2
        self.b2 += learning_rate * db2

    def fit(self, X, epochs, learning_rate):
        """
        Trains the autoencoder on the given data.
        
        Parameters:
        X (numpy.ndarray): Training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, output, learning_rate)
            
    #Added this to make the testing functions more simplistic        
    def predict(self, X): 
        """
        Utilizes the Forward method to Predict the output using the trained autoencoder. Added to make testing functions simpler before I split them out into seperate files
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Reconstructed output.
        """
        return self.forward(X)

class CombinedNetwork:
    """
    A combined network using an autoencoder for feature extraction and a feedforward network for prediction.
    
    Attributes:
        autoencoder (Autoencoder): Trained autoencoder.
        hidden_size (int): Number of hidden nodes in the additional hidden layer.
        output_size (int): Number of output nodes.
        W3 (numpy.ndarray): Weights between autoencoder's hidden layer and additional hidden layer.
        b3 (numpy.ndarray): Biases for additional hidden layer.
        W4 (numpy.ndarray): Weights between additional hidden layer and output layer.
        b4 (numpy.ndarray): Biases for output layer.
    """
    def __init__(self, autoencoder, hidden_size, output_size):
        """
        Initializes the combined network with given autoencoder, hidden layer size, and output layer size.
        
        Parameters:
        autoencoder (Autoencoder): Trained autoencoder.
        hidden_size (int): Number of hidden nodes in the additional hidden layer.
        output_size (int): Number of output nodes.
        """
        self.autoencoder = autoencoder
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W3 = np.random.randn(autoencoder.hidden_size, hidden_size)
        self.b3 = np.zeros(hidden_size)
        self.W4 = np.random.randn(hidden_size, output_size)
        self.b4 = np.zeros(output_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        """
        Performs the forward pass of the combined network.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Network output.
        """
        self.a1 = self.autoencoder.sigmoid(np.dot(X, self.autoencoder.W1) + self.autoencoder.b1)
        self.z3 = np.dot(self.a1, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = self.sigmoid(self.z4)
        return self.a4

    def backward(self, X, y, learning_rate):
        """
        Performs the backward pass and updates weights and biases.
        
        Parameters:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): Target labels.
        learning_rate (float): Learning rate for weight updates.
        """
        output = self.forward(X)
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.W4.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a3)

        self.W4 += np.dot(self.a3.T, output_delta) * learning_rate
        self.b4 += np.sum(output_delta, axis=0) * learning_rate
        self.W3 += np.dot(self.a1.T, hidden_delta) * learning_rate
        self.b3 += np.sum(hidden_delta, axis=0) * learning_rate

    def fit(self, X, y, epochs, learning_rate):
        """
        Trains the combined network on the given data.
        
        Parameters:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Target labels.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        """
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y - self.forward(X)) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

    #Added this to make the testing functions more simplistic        
    def predict(self, X):
        """
        Predicts the output using the trained combined network. Added to make testing functions simpler before I split them out into seperate files.
        
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
    y = breast_cancer['Class'].values.reshape(-1, 1)
    X = breast_cancer.drop(columns=['ID', 'Class']).values
    X_breast_cancer_normalized = min_max_normalization(X)

    X_breast, X_test, y_breast, y_test = train_test_split(X_breast_cancer_normalized, y, test_size=0.2)

    # Train autoencoder on each dataset
    autoencoder_breast = Autoencoder(input_size=X_breast.shape[1], hidden_size=10)
    autoencoder_breast.fit(X_breast, epochs=1000, learning_rate=0.1)

    # Create combined networks
    combined_network_breast = CombinedNetwork(autoencoder_breast, hidden_size=10, output_size=1)
    combined_network_breast.fit(X_breast, y_breast, epochs=1000, learning_rate=0.1)

    print("Breast Cancer Dataset Performance:")
    evaluate_classification(combined_network_breast, X_test, y_test)

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

    autoencoder_abalone = Autoencoder(input_size=X_train.shape[1], hidden_size=10)
    autoencoder_abalone.fit(X_train, epochs=1000, learning_rate=0.001)

    combined_network_abalone = CombinedNetwork(autoencoder_abalone, hidden_size=10, output_size=1)
    combined_network_abalone.fit(X_train, y_train, epochs=1000, learning_rate=0.001)

    print("Abalone Dataset Performance:")
    evaluate_regression(combined_network_abalone, X_test, y_test)
