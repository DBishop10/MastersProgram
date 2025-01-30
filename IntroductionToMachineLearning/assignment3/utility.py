import pandas as pd
import numpy as np

def encode_categorical_features(df):
    """
    Encodes categorical features in the dataframe using one-hot encoding.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the features to be encoded.
    
    Returns:
    df_encoded (pandas.DataFrame): DataFrame with categorical features encoded.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return df_encoded

def fill_missing_values(df):
    """
    Fills in missing values in dataframes

    Parameters:
    df (pandas.DataFrame): DataFrame with categorical features encoded.

    Returns:
    df (pandas.DataFrame): DataFrame with categorical features encoded.
    """
    for column in df.columns:
        if df[column].dtype == np.number:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def train_test_split(X, y, test_size=0.2, stratify=False):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Labels array.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.2
    stratify (bool): Whether to stratify the split based on the labels. Default is false
    
    Returns:
    tuple: Tuple containing training and testing splits of X and y.
    """
    if stratify:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        train_indices, test_indices = [], []

        for cls, count in zip(unique_classes, class_counts):
            indices = np.where(y == cls)[0]
            np.random.shuffle(indices)
            split_point = int(count * (1 - test_size))
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])
    else:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        split_point = int(len(y) * (1 - test_size))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy score of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    int: The mean of true vs predicted y values
    """
    return np.mean(y_true == y_pred)

def mean_squared_error(y_true, y_pred):
    """
    Calculates the accuracy score of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    int: The mean of true vs predicted y values squared
    """
    return np.mean((y_true - y_pred) ** 2)

def min_max_normalization(X):
    """
    Applies min-max normalization to the dataset.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    
    Returns:
    numpy.ndarray: Normalized feature matrix.
    """
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val)

def one_hot_encode(y, num_classes):
    """
    Encodes categorical data utilizing one_hot_encode

    Parameters:
    y (numpy.ndarray): Actual y values
    num_classes (int): Number of classes

    Return:
    one_hot (numpy.ndarray): encoded version of y values
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def precision_score(y_true, y_pred):
    """
    Calculates the precision score of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    int: the amount of True positives divided by true_positives + false positives
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def recall_score(y_true, y_pred):
    """
    Calculates the recall score of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    int: the amount of True positives divided by true_positives + false negatives
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def f1_score(y_true, y_pred):
    """
    Calculates the f1 score of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    int: calculated f1 score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred):
    """
    Creates the confusion matrix of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    np.array: of true positive, false positive, false negative, and true negative
    """
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[true_positive, false_positive],
                     [false_negative, true_negative]])

def r_squared(y_true, y_pred):
    """
    Calculates the r^2 of the model

    Parameters:
    y_true (numpy.ndarray): Actual y values
    y_pred (numpy.ndarray): Predicted y values

    Return:
    int: r^2 of the model
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_classification(model, X, y):
    """
    Evaluates the classification model and returns various metrics.
    
    Parameters:
    model: The trained classification model.
    X (numpy.ndarray): Test features.
    y (numpy.ndarray): True labels.
    
    Returns:
    dict: Dictionary containing accuracy, precision, recall, f1 score, and confusion matrix.
    """
    predictions = model.predict(X)
    y_pred = (predictions > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f'Accuracy: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{cm}')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def evaluate_regression(model, X, y):
    """
    Evaluates the regression model and returns various metrics.
    
    Parameters:
    model: The trained regression model.
    X (numpy.ndarray): Test features.
    y (numpy.ndarray): True labels.
    
    Returns:
    dict: Dictionary containing mean squared error and R-squared score.
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r_squared(y, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'R-Squared: {r2}')
    return {
        'mse': mse,
        'r_squared': r2
    }

def encode_labels(y):
    """
    Encodes labels into numeric values.
    
    Parameters:
    y (numpy.ndarray or list): Array or list of labels.
    
    Returns:
    tuple: Tuple containing the encoded labels array and a dictionary mapping original labels to encoded values.
    """
    unique_classes = np.unique(y)
    label_to_index = {label: index for index, label in enumerate(unique_classes)}
    encoded_y = np.array([label_to_index[label] for label in y])
    return encoded_y, label_to_index