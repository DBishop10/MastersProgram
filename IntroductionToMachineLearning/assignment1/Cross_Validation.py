import numpy as np
from Nonparametric import *
from Generate_Graphs import plot_confusion_matrix, plot_regression_results, confusion_matrix

def split_data(X, y, test_size=0.2, stratify=False):
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

def stratified_k_fold_split(X, y, k_folds):
    """
    Performs stratified k-fold split on the dataset.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Labels array.
    k_folds (int): Number of folds.
    
    Returns:
    list: List of fold indices.
    """
    unique_classes, y_indices = np.unique(y, return_inverse=True)
    fold_indices = [[] for _ in range(k_folds)]

    for class_index in range(len(unique_classes)):
        class_indices = np.where(y_indices == class_index)[0]
        np.random.shuffle(class_indices)
        folds = np.array_split(class_indices, k_folds)
        for fold_index, fold in enumerate(folds):
            fold_indices[fold_index].extend(fold)

    return fold_indices

def stratified_split(X, y):
    """
    Splits the dataset into two equal parts, stratified by class labels.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Labels array.
    
    Returns:
    tuple: Tuple containing two arrays of indices for the splits.
    """
    unique_classes, y_indices = np.unique(y, return_inverse=True)
    split_indices = []

    for class_index in range(len(unique_classes)):
        class_indices = np.where(y_indices == class_index)[0]
        np.random.shuffle(class_indices)
        split_indices.append(class_indices)

    split_1_indices, split_2_indices = [], []
    for indices in split_indices:
        midpoint = len(indices) // 2
        split_1_indices.extend(indices[:midpoint])
        split_2_indices.extend(indices[midpoint:])

    return np.array(split_1_indices), np.array(split_2_indices)

def k_fold_split(X, k_folds):
    """
    Performs k-fold split on the dataset.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    k_folds (int): Number of folds.
    
    Returns:
    list: List of fold indices.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, len(X) // k_folds, dtype=int)
    fold_sizes[:len(X) % k_folds] += 1
    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        fold_indices.append(indices[current:current + fold_size])
        current += fold_size
    return fold_indices

def generate_param_combinations(param_grid):
    """
    Generates all combinations of hyperparameters from the given parameter grid.
    
    Parameters:
    param_grid (dict): Dictionary where keys are hyperparameter names and values are lists of possible values.
    
    Returns:
    list: List of dictionaries, each containing a unique combination of hyperparameters.
    """
    import itertools
    all_params = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    return all_params

def tune_hyperparameters(model_class, X_train, y_train, param_grid, mode='classification'):
    """
    Tunes hyperparameters using k-fold cross-validation.
    
    Parameters:
    model_class (class): Model class to be evaluated.
    param_grid (dict): Dictionary where keys are hyperparameter names and values are lists of possible values.
    X_train (numpy.ndarray): Training feature matrix.
    y_train (numpy.ndarray): Training labels array.
    mode (str): Type of task ('classification' or 'regression').
    
    Returns:
    dict: Best hyperparameters found.
    """
    best_params = None
    best_score = -np.inf if mode == 'classification' else np.inf

    param_combinations = generate_param_combinations(param_grid)

    for params in param_combinations:
        scores = []
        for _ in range(5):
            split_1_indices, split_2_indices = stratified_split(X_train, y_train)
            X1, X2 = X_train[split_1_indices], X_train[split_2_indices]
            y1, y2 = y_train[split_1_indices], y_train[split_2_indices]
            model_1 = model_class(**{k: (int(v) if k == 'k' else v) for k, v in params.items()})
            model_2 = model_class(**{k: (int(v) if k == 'k' else v) for k, v in params.items()})
            model_1.fit(X1, y1)
            model_2.fit(X2, y2)
            preds_1_on_2 = model_1.predict(X2)
            preds_2_on_1 = model_2.predict(X1)
            if mode == 'classification':
                score_1_on_2 = np.sum(preds_1_on_2 == y2) / len(y2)
                score_2_on_1 = np.sum(preds_2_on_1 == y1) / len(y1)
            elif mode == 'regression':
                score_1_on_2 = np.mean((preds_1_on_2 - y2) ** 2)
                score_2_on_1 = np.mean((preds_2_on_1 - y1) ** 2)
            else:
                raise ValueError("Invalid mode. Choose 'classification' or 'regression'.")
            scores.append(score_1_on_2)
            scores.append(score_2_on_1)

        mean_score = np.mean(scores)

        if (mode == 'classification' and mean_score > best_score) or (mode == 'regression' and mean_score < best_score):
            best_score = mean_score
            best_params = params

    return best_params

def cross_validation(model, X, y, k_folds=5, mode='classification', stratify=False):
    """
    Performs k-fold cross-validation and returns the mean score.
    
    Parameters:
    model (object): Model instance to be evaluated.
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Labels array.
    k_folds (int): Number of folds.
    mode (str): Type of task ('classification' or 'regression').
    
    Returns:
    tuple: Mean score and list of scores for each fold.
    """
    fold_indices = stratified_k_fold_split(X, y, k_folds) if stratify and mode == 'classification' else k_fold_split(X, k_folds)
    scores = []

    for fold in range(k_folds):
        test_indices = fold_indices[fold]
        train_indices = np.setdiff1d(np.arange(len(X)), test_indices)

        if len(test_indices) == 0 or len(train_indices) == 0:
            continue

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if mode == 'classification':
            score = np.sum(predictions == y_test) / len(y_test)
        elif mode == 'regression':
            score = np.mean((predictions - y_test) ** 2)
        else:
            raise ValueError("Invalid mode. Choose 'classification' or 'regression'.")

        scores.append(score)

    mean_score = np.mean(scores)
    return mean_score, scores

def final_evaluation(model_class, X_train, y_train, X_test, y_test, best_params, mode='classification', datasetname=""):
    """
    Performs final evaluation of the model using a stratified split and returns the mean score and test score.
    
    Parameters:
    model_class (class): Model class to be evaluated.
    X_train (numpy.ndarray): Training feature matrix.
    y_train (numpy.ndarray): Training labels array.
    X_test (numpy.ndarray): Testing feature matrix.
    y_test (numpy.ndarray): Testing labels array.
    best_params (dict): Best parameters for the model.
    mode (str): Type of task ('classification' or 'regression').
    datasetname (str): What dataset it is doing the evaluation on

    Returns:
    tuple: Mean score and test score.
    """
    scores = []

    for _ in range(5):
        split_1_indices, split_2_indices = stratified_split(X_train, y_train)
        X1, X2 = X_train[split_1_indices], X_train[split_2_indices]
        y1, y2 = y_train[split_1_indices], y_train[split_2_indices]

        model_1 = model_class(**best_params)
        model_2 = model_class(**best_params)

        model_1.fit(X1, y1)
        model_2.fit(X2, y2)

        preds_1_on_2 = model_1.predict(X2)
        preds_2_on_1 = model_2.predict(X1)

        if mode == 'classification':
            score_1_on_2 = np.sum(preds_1_on_2 == y2) / len(y2)
            score_2_on_1 = np.sum(preds_2_on_1 == y1) / len(y1)
        elif mode == 'regression':
            score_1_on_2 = np.mean((preds_1_on_2 - y2) ** 2)
            score_2_on_1 = np.mean((preds_2_on_1 - y1) ** 2)
        else:
            raise ValueError("Invalid mode. Choose 'classification' or 'regression'.")

        scores.append(score_1_on_2)
        scores.append(score_2_on_1)

    # Evaluate on the test set with the best model
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)
    test_predictions = best_model.predict(X_test)
    title_suffix = model_class.__name__ + ' - '.join(f"{key}={val}" for key, val in best_params.items()) + datasetname

    if mode == 'classification':
        test_score = np.sum(test_predictions == y_test) / len(y_test)
        cm = confusion_matrix(y_test, test_predictions, len(np.unique(y_test)))
        plot_confusion_matrix(cm, classes=np.unique(y_test), filename=f'confusion_matrix_{title_suffix}.png', datasetname=datasetname)
    elif mode == 'regression':
        test_score = np.mean((test_predictions - y_test) ** 2)
        plot_regression_results(y_test, test_predictions, filename=f'regression_plot_{title_suffix}.png', datasetname=datasetname)
    else:
        raise ValueError("Invalid mode. Choose 'classification' or 'regression'.")

    return np.mean(scores), test_score