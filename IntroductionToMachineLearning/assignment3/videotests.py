import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import *
from regressionmodels import *
from feedforward_nn import *
from autoencoder import *
from combined_network_testing import *
from feedforward_nn_testing import *
from regressionmodels_testing import *

def train_and_evaluate(model, X_train, y_train, X_test, y_test, task='classification'):
    """
    This is a one fold form of cross validation, just removed the cross validation part for the video.

    Parameters:
    model (any model type): The model that you are training
    X_train (numpy.ndarray): X values to train on
    y_train (numpy.ndarray): y values to train on
    X_test (numpy.ndarray): X values to test performance
    y_test (numpy.ndarray): y values to test performance
    task (string): Whether it is classification or regression
    """
    if task == 'classification':
        if isinstance(model, FeedforwardNeuralNetwork) or isinstance(model, CombinedNetwork):
            y_train_encoded = one_hot_encode(y_train, num_classes=2)
            model.fit(X_train, y_train_encoded, epochs=101, learning_rate=0.01)
        else:
            model.fit(X_train, y_train)
        
        if isinstance(model, FeedforwardNeuralNetwork) or isinstance(model, CombinedNetwork):
            y_test_encoded = one_hot_encode(y_test, num_classes=2)
            evaluate_classification(model, X_test, y_test_encoded)
        else:
            evaluate_classification(model, X_test, y_test)
    else:
        if isinstance(model, LinearRegression) or isinstance(model,LogisticRegression):
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, epochs=10, learning_rate=0.01)
        evaluate_regression(model, X_test, y_test)

def demonstrate_weight_updates(model):
    """
    Print out the weights of the models to show updates

    Parameters:
    model (any model type): Model you want to see the weights of
    """
    if isinstance(model, FeedforwardNeuralNetwork):
        print("Weights:", model.weights)
        print("Biases:", model.biases)
    elif isinstance(model, CombinedNetwork):
        print("Autoencoder W1:", model.autoencoder.W1)
        print("Autoencoder b1:", model.autoencoder.b1)
        print("Autoencoder W2:", model.autoencoder.W2)
        print("Autoencoder b2:", model.autoencoder.b2)
        print("Combined W3:", model.W3)
        print("Combined b3:", model.b3)
        print("Combined W4:", model.W4)
        print("Combined b4:", model.b4)


def demonstrate_propagation_ffnn(model, X_example):
    """
    Demonstrates how an example is propagated through a feedforward neural network.
    
    Parameters:
    model (FeedforwardNeuralNetwork): The feedforward neural network model.
    X_example (numpy.ndarray): The example input.
    
    Returns:
    None
    """
    layer_inputs = []
    layer_outputs = [X_example]

    # Propagate through each layer
    for i in range(len(model.weights)):
        input_to_layer = np.dot(layer_outputs[-1], model.weights[i]) + model.biases[i]
        layer_inputs.append(input_to_layer)

        if i == len(model.weights) - 1:
            output = model.output_activation_function(input_to_layer)
        else:
            output = model.activation_function(input_to_layer)

        layer_outputs.append(output)

    # Print the propagation details
    print("Example Propagation through Feedforward Neural Network:")
    print("Input:", X_example)
    for i in range(len(model.weights)):
        print(f"Layer {i + 1} Input:", layer_inputs[i])
        print(f"Layer {i + 1} Output:", layer_outputs[i + 1])
    print("Output:", layer_outputs[-1])


def demonstrate_propagation_combined_network(model, X_example):
    """
    Demonstrates how an example is propagated through a combined network.
    
    Parameters:
    model (CombinedNetwork): The combined network model.
    X_example (numpy.ndarray): The example input.
    
    Returns:
    None
    """
    # Propagation through the autoencoder part
    Z1 = np.dot(X_example, model.autoencoder.W1) + model.autoencoder.b1
    A1 = model.autoencoder.sigmoid(Z1)
    
    # Propagation through the additional layers
    Z2 = np.dot(A1, model.W3) + model.b3
    A2 = model.autoencoder.sigmoid(Z2)
    Z3 = np.dot(A2, model.W4) + model.b4
    A3 = model.autoencoder.sigmoid(Z3)
    
    print("Example Propagation through Combined Network:")
    print("Input:", X_example)
    print("Autoencoder Encoding Layer Activations:", A1)
    print("Combined Network Layer 1 Activations:", A2)
    print("Output:", A3)

if __name__ == "__main__":
    datasets = {}
    breast_cancer_path = '../Data/breastcancer/breast-cancer-wisconsin.data'
    breast_cancer_columns = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                             'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
                             'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
    breast_cancer = pd.read_csv(breast_cancer_path, header=None, names=breast_cancer_columns, na_values='?')
    breast_cancer.fillna(breast_cancer.median(), inplace=True)
    breast_cancer['Bare_Nuclei'] = pd.to_numeric(breast_cancer['Bare_Nuclei'], errors='coerce')
    breast_cancer.fillna(breast_cancer.median(), inplace=True)
    breast_cancer['Class'] = breast_cancer['Class'].apply(lambda x: 1 if x == 4 else 0)
    y_breast = breast_cancer['Class'].values
    X_breast = breast_cancer.drop(columns=['ID', 'Class']).values
    X_breast = min_max_normalization(X_breast)
    datasets['Breast Cancer'] = (X_breast, y_breast, 'classification')

    abalone_path = '../Data/abalone/abalone.data'
    abalone_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
                      'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    abalone = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    abalone['Sex'] = abalone['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    y_abalone = abalone['Rings'].values.reshape(-1, 1)
    X_abalone = abalone.drop(columns=['Rings']).values
    X_abalone = min_max_normalization(X_abalone)
    y_normalized = min_max_normalization(y_abalone)
    datasets['Abalone'] = (X_abalone, y_normalized, 'regression')

    for dataset_name, (X, y, task) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=(task == 'classification'))

        if task == 'classification':
            model = LogisticRegression(learning_rate=0.01, num_iterations=10, debug=True)

            print(f"\n{model.__class__.__name__} ({dataset_name})")

            input("")

            print("Fold One")
            train_and_evaluate(model, X_train, y_train, X_test, y_test, task)

            input("")
            if task == 'classification':
                model = FeedforwardNeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[10, 10], output_size=2, activation='tanh', output_activation='softmax', loss='cross_entropy')
            else:
                model = FeedforwardNeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[10, 10], output_size=1, activation='tanh', output_activation='linear', loss='mse')
            
            print(f"\n{model.__class__.__name__} ({dataset_name})")
            input("")

            demonstrate_weight_updates(model)
            input("Weights Start")

            print("Fold One")
            train_and_evaluate(model, X_train, y_train, X_test, y_test, task)

            input("")

            demonstrate_propagation_ffnn(model, X_train[0:1])

            input("Demonstrate Propogation")

            demonstrate_weight_updates(model)
            input("Weight Updates")

            autoencoder = Autoencoder(input_size=X_train.shape[1], hidden_size=10)

            if task == 'classification':
                model = CombinedNetwork(autoencoder, hidden_size=10, output_size=2)
            else:
                model = CombinedNetwork(autoencoder, hidden_size=10, output_size=1)

            print(f"\n{model.__class__.__name__} ({dataset_name})")
            input("")

            demonstrate_weight_updates(model)
            input("Weights Start")

            demonstrate_propagation_combined_network(model, X_train[0:1])
            
            input("Demonstrate Propogation")

            X_example = np.array([[0.5, 0.2, 0.1]])
            print(X_example)
            autoencoder = Autoencoder(input_size=3, hidden_size=2, debug=True)
            autoencoder.fit(X_example, epochs=1, learning_rate=.00000001)

            input("Encode Decode")

            print("Fold One")
            train_and_evaluate(model, X_train, y_train, X_test, y_test, task)
            model.autoencoder.fit(X_train[0:1], epochs=1000, learning_rate=.1)

            demonstrate_weight_updates(model)

            input("Weight Updates")
        else:
            model = LinearRegression(learning_rate=0.01, num_iterations=10, debug=True)

            print(f"\n{model.__class__.__name__} ({dataset_name})")

            input("")

            print("Fold One")
            train_and_evaluate(model, X_train, y_train, X_test, y_test, task)

            input("")
