import numpy as np
import pandas as pd
from Nonparametric import KNN, EditedKNN, calculate_distances, gaussian_kernel, calculate_distances
from Cross_Validation import split_data, stratified_split
import matplotlib.pyplot as plt
import os
from Tests import z_score_normalization, min_max_normalization, encode_categorical_features
# Example data for demonstration
def load_abalone_data(preencode=True):
    abalone_path = '../Data/abalone/abalone.data'
    abalone_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    abalone_df = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    if(preencode):
        abalone_df_encoded = encode_categorical_features(abalone_df)
        X = abalone_df_encoded.drop(columns=['Rings']).values
        y = abalone_df_encoded['Rings'].values
    else:
        X = abalone_df.drop(columns=['Rings'])
        y = abalone_df['Rings']
    return X, y

# Demonstration of z-score normalization
def demonstrate_z_score_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    print("Mean (μ):", mean)
    print("Standard Deviation (σ):", std)
    print("First 5 rows of normalized data:\n", X_normalized[:5])

# Demonstration of categorical encoding
def demonstrate_categorical_encoding():
    X, _ = load_abalone_data(preencode=False)
    print("First 5 rows of categorical data before encoding:\n", X[:5])
    X_encoded = encode_categorical_features(X)
    print("First 5 rows of categorical data after encoding:\n", X_encoded[:5])

# Demonstration of 5x2 cross-validation split
def demonstrate_cross_validation_split(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=False)
    split_1_indices, split_2_indices = stratified_split(X_train, y_train)
    print("Train set size:", (len(X_train) + len(y_train)), "Test set size:", (len(X_test) + len(y_test)))
    print("Split 1 size:", len(split_1_indices), "Split 2 size:", len(split_2_indices))

# Demonstration of distance calculation
def demonstrate_distance_calculation(X):
    distances = calculate_distances(X[:2], X[:2], 'euclidean')
    print("Distance matrix:\n", distances)

# Demonstration of kernel calculation
def demonstrate_kernel_calculation(X):
    distances = calculate_distances(X[:2], X[:2], 'euclidean')
    kernel_values = gaussian_kernel(distances, 0.1)
    print("Kernel values:\n", kernel_values)

# Demonstration of k-NN classification
def demonstrate_knn_classification(X, y):
    knn = KNN(k=3, mode='classification')
    knn.fit(X, y)
    point = X[0]
    neighbors = knn._predict(calculate_distances(np.array([point]), X))
    print("Neighbors for classification of point:", neighbors)

# Demonstration of k-NN regression
def demonstrate_knn_regression(X, y):
    knn = KNN(k=3, mode='regression', gamma=0.1)
    knn.fit(X, y)
    point = X[0]
    neighbors = knn._predict(calculate_distances(np.array([point]), X))
    print("Neighbors for regression of point:", neighbors)

# Demonstration of edited nearest neighbor
def demonstrate_edited_knn(X, y):
    edited_knn = EditedKNN(k=3, mode='classification', epsilon=0.1)
    edited_knn.fit(X, y)
    edited_X, edited_y = edited_knn.edit_training_set(X, y)
    print("Original training set size:", len(X))
    print("Edited training set size:", len(edited_X))

def main():
    X, y = load_abalone_data()

    # Normalize the data using z-score
    demonstrate_z_score_normalization(X)
    input("This is to pause the code")

    # Encode categorical features
    demonstrate_categorical_encoding()
    input("This is to pause the code")
    
    # Split the data for cross-validation
    demonstrate_cross_validation_split(X, y)
    input("This is to pause the code")

    # Calculate distances
    demonstrate_distance_calculation(X)
    input("This is to pause the code")

    # Calculate kernel values
    demonstrate_kernel_calculation(X)
    input("This is to pause the code")

    # Classify a point using k-NN
    demonstrate_knn_classification(X, y)
    input("This is to pause the code")

    # Regress a point using k-NN
    demonstrate_knn_regression(X, y)
    input("This is to pause the code")

    # Demonstrate edited nearest neighbor
    demonstrate_edited_knn(X, y)
    input("This is to pause the code")

if __name__ == "__main__":
    main()