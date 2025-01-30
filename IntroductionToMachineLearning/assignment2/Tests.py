import numpy as np
from DecisionTree import *
from GenerateGraphs import * 

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

# Custom label encoder for target values
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

def cross_val_metrics(model, X, y, cv=2, repeats=5, task='classification'):
    """
    Performs cross-validation and computes evaluation metrics.

    Args:
        model: The model to be evaluated.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        cv (int): Number of cross-validation folds. Default is 2.
        repeats (int): Number of times to repeat cross-validation. Default is 5.
        task (str): Task type, either 'classification' or 'regression'. Default is 'classification'.
        prune (boolean): Whether or not we are testing a pruned model or a non pruned model.

    Returns:
        numpy.ndarray: Mean evaluation metrics across all cross-validation runs.
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    all_scores = []

    for repeat in range(repeats):
        np.random.shuffle(indices)
        fold_size = n_samples // cv

        for i in range(cv):
            val_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            model.fit(X_train, y_train)
            
            if task == 'classification':
                accuracy, precision, recall, f1 = evaluate_classification_model(model, X_val, y_val)
                scores = (accuracy, precision, recall, f1)
            else:
                mse, r2 = evaluate_regression_model(model, X_val, y_val)
                scores = (mse, r2)

            # Get an example fold for video
            if repeat == 0 and i == 0:
                print(f"First Fold (Repeat {repeat+1}, Fold {i+1}):")
                print("Predictions:", model.predict(X_val)[:5])
                print("Actual:", y_val[:5])
                if task == 'classification':
                    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                else:
                    print(f"Mean Squared Error: {mse:.4f}, R-Squared: {r2:.4f}")
                print("="*50)
            
            all_scores.append(scores)

    all_scores = np.array(all_scores)
    return np.mean(all_scores, axis=0)

# Evaluate models
def evaluate_classification_model(model, X, y):
    """
    Evaluates a classification model and computes performance metrics.

    Args:
        model: The classification model to be evaluated.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): True labels.

    Returns:
        tuple: Evaluation metrics (accuracy, average precision, average recall, average F1 score).
    """
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    
    unique_classes = np.unique(y)
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for cls in unique_classes:
        tp = np.sum((preds == cls) & (y == cls))
        fp = np.sum((preds == cls) & (y != cls))
        fn = np.sum((preds != cls) & (y == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Calculate the average precision, recall, and f1-score
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    
    return accuracy, avg_precision, avg_recall, avg_f1

def evaluate_regression_model(model, X, y):
    """
    Evaluates a regression model and computes performance metrics.

    Args:
        model: The regression model to be evaluated.
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): True labels.

    Returns:
        tuple: Evaluation metrics (mean squared error, R-squared).
    """
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)
    r2 = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return mse, r2

# Load Datasets
def load_datasets():
    """
    Loads various datasets for classification and regression tasks.

    Returns:
        dict: Dictionary containing datasets with feature matrices and labels.
    """
    datasets = {}

    # Abalone dataset (Regression)
    abalone_path = '../Data/abalone/abalone.data'
    abalone_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    abalone_df = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    X_abalone = abalone_df.drop(columns=['Rings']).values
    y_abalone = abalone_df['Rings'].values
    datasets['Abalone'] = (X_abalone, y_abalone, 'regression')

    # Breast Cancer dataset (Classification)
    breast_cancer_path = '../Data/breastcancer/breast-cancer-wisconsin.data'
    breast_cancer_columns = [
        'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
        'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses', 'Class'
    ]
    breast_cancer_df = pd.read_csv(breast_cancer_path, header=None, names=breast_cancer_columns)
    breast_cancer_df.replace('?', np.nan, inplace=True)
    breast_cancer_df['Bare Nuclei'] = breast_cancer_df['Bare Nuclei'].astype(float)
    breast_cancer_df.fillna(breast_cancer_df.mean().iloc[0], inplace=True)
    X_breast_cancer = breast_cancer_df.drop(columns=['Sample code number', 'Class']).values
    y_breast_cancer = breast_cancer_df['Class'].values
    datasets['Breast Cancer'] = (X_breast_cancer, y_breast_cancer, 'classification')

    # Car Evaluation dataset (Classification)
    car_path = '../Data/carevaluation/car.data'
    car_columns = [
        'Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class'
    ]
    car_df = pd.read_csv(car_path, header=None, names=car_columns)
    y_car = car_df['Class'].values
    X_car = car_df.drop(columns=['Class']).values
    y_car, car_label_mapping = encode_labels(y_car)
    datasets['Car Evaluation'] = (X_car, y_car, 'classification')

    # Machine dataset (Regression)
    machine_path = '../Data/computerhardware/machine.data'
    machine_columns = [
        'Vendor name', 'Model name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'
    ]
    machine_df = pd.read_csv(machine_path, header=None, names=machine_columns)
    y_machine = machine_df['PRP'].values
    X_machine = machine_df.drop(columns=['PRP']).values
    datasets['Machine'] = (X_machine, y_machine, 'regression')

    # House Votes 84 dataset (Classification)
    house_votes_path = '../Data/congressionalvotingrecords/house-votes-84.data'
    house_votes_columns = [
        'Class name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
        'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 
        'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'
    ]
    house_votes_df = pd.read_csv(house_votes_path, header=None, names=house_votes_columns)
    house_votes_df.replace('?', np.nan, inplace=True)
    house_votes_df.fillna(house_votes_df.mode().iloc[0], inplace=True)
    y_house_votes = house_votes_df['Class name'].values
    y_house_votes, house_votes_label_mapping = encode_labels(y_house_votes)
    X_house_votes = house_votes_df.drop(columns=['Class name']).values
    datasets['House Votes 84'] = (X_house_votes, y_house_votes, 'classification')

    # Forest Fires dataset (Regression)
    forest_fires_path = '../Data/forestfires/forestfires.csv'
    forest_fires_df = pd.read_csv(forest_fires_path)
    forest_fires_df['log_area'] = np.log(forest_fires_df['area'] + 1)
    X_forest_fires = forest_fires_df.drop(['area', 'log_area'], axis=1).values
    y_forest_fires = forest_fires_df['log_area'].values
    datasets['Forest Fires'] = (X_forest_fires, y_forest_fires, 'regression')

    return datasets

def main():
    datasets = load_datasets()

    for dataset_name, (X, y, task) in datasets.items():
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, stratify=(task == 'classification'))
        
        if task == 'classification':
            model = DecisionTreeClassifier(max_depth=10)
            graph_function = save_classification_graph
        else:
            model = DecisionTreeRegressor(max_depth=10)
            graph_function = save_regression_graph

        model.fit(X_train, y_train)

        # Evaluate before pruning
        metrics_before = cross_val_metrics(model, X_train, y_train, cv=5, task=task)

        # Prune the model
        model.prune(X_train, y_train)

        # Evaluate after pruning
        metrics_after = cross_val_metrics(model, X_train, y_train, cv=5, task=task)

        # Create Metric Graph
        graph_function(metrics_before, metrics_after, dataset_name)

if __name__ == "__main__":
    main()