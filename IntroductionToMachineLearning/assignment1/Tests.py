import pandas as pd
from Cross_Validation import * 

#Normalization Methods
def z_score_normalization(X):
    """
    Applies z-score normalization to the dataset.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    
    Returns:
    numpy.ndarray: Normalized feature matrix.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

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

# Simplistic encoding for categorical features, utilizes pandas get_dummies which I think is okay but can be done without it if requested
def encode_categorical_features(df):
    """
    Encodes categorical features in the dataframe using one-hot encoding.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the features to be encoded.
    
    Returns:
    pandas.DataFrame: DataFrame with categorical features encoded.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return df_encoded

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

if __name__ == "__main__":
    # Define parameter grid for hyperparameter tuning, hoping to add more and get a better parameter find
    regression_param_grid = {
        'k': [1, 3, 5, 7, 9],
        'gamma': [0.01, 0.05, 0.1, 0.5, 1, 10],
        'distance_metric': ['euclidean', 'manhattan']
    }

    edited_regression_param_grid = {
        'k': [1, 3, 5, 7, 9],
        'gamma': [0.01, 0.05, 0.1, 0.5, 1, 10],
        'epsilon': [0.01, 0.05, 0.1, 0.5, 1],
        'distance_metric': ['euclidean', 'manhattan']
    }

    classification_param_grid = {
        'k': [1, 3, 5, 7, 9],
        'distance_metric': ['euclidean', 'manhattan']
    }

    edited_classification_param_grid = {
        'k': [1, 3, 5, 7, 9],
        'epsilon': [0.01, 0.05, 0.1, 0.5, 1],
        'distance_metric': ['euclidean', 'manhattan']
    }
    #Place all test data into a text file
    f = open("Tests.txt", "a")

    def perform_tests(dataset_name, X, y):
        print(f"Testing on {dataset_name} Dataset")
        f.write(f"Testing on {dataset_name} Dataset\n")

        X_train_Clas, X_test_Clas, y_train_Clas, y_test_Clas = split_data(X, y, stratify=True)

        # Classification KNN
        print("Starting Classification KNN")
        f.write("Starting Classification KNN\n")
        best_params = tune_hyperparameters(KNN, X_train_Clas, y_train_Clas, classification_param_grid, mode='classification')
        f.write(f"Best hyperparameters: {best_params}\n")
        mean_score, scores = cross_validation(KNN(**best_params), X_train_Clas, y_train_Clas, k_folds=5, mode='classification')
        f.write(f"Mean cross-validation accuracy on training data: {mean_score}\n")
        final_mean_score, final_test_score = final_evaluation(KNN, X_train_Clas, y_train_Clas, X_test_Clas, y_test_Clas, best_params, mode='classification', datasetname=dataset_name)
        f.write(f"Final mean accuracy on cross-validation: {final_mean_score}\n")
        f.write(f"Final mean accuracy on test data: {final_test_score}\n")


        # Classification Edited KNN
        print("Starting Classification Edited KNN")
        f.write("Starting Classification Edited KNN\n")
        best_edited_classification_params = tune_hyperparameters(EditedKNN, X_train_Clas, y_train_Clas, edited_classification_param_grid, mode='classification')
        f.write(f"Best hyperparameters for EditedKNN classification: {best_edited_classification_params}\n")
        mean_score, scores = cross_validation(EditedKNN(**best_edited_classification_params), X_train_Clas, y_train_Clas, k_folds=5, mode='classification')
        f.write(f"Mean cross-validation accuracy on training data for EditedKNN classification: {mean_score}\n")
        final_mean_score, final_test_score = final_evaluation(EditedKNN, X_train_Clas, y_train_Clas, X_test_Clas, y_test_Clas, best_edited_classification_params, mode='classification', datasetname=dataset_name)
        f.write(f"Final mean accuracy on cross-validation for EditedKNN classification: {final_mean_score}\n")
        f.write(f"Final mean accuracy on test data for EditedKNN classification: {final_test_score}\n")

        f.write("\n\n\n")

    def perform_tests_regression(dataset_name, X, y):
        print(f"Testing on {dataset_name} Dataset")
        f.write(f"Testing on {dataset_name} Dataset\n")

        X_train, X_test, y_train, y_test = split_data(X, y, stratify=False)

        # Regression KNN
        print("Starting Regression KNN")
        f.write("Starting Regression KNN\n")
        best_params = tune_hyperparameters(KNN, X_train, y_train, regression_param_grid, mode='regression')
        f.write(f"Best hyperparameters: {best_params}\n")
        mean_score, scores = cross_validation(KNN(**best_params), X_train, y_train, k_folds=5, mode='regression', stratify=False)
        f.write(f"Mean cross-validation score on training data: {mean_score}\n")
        final_mean_score, final_test_score = final_evaluation(KNN, X_train, y_train, X_test, y_test, best_params, mode='regression', datasetname=dataset_name)
        f.write(f"Final mean score on cross-validation: {final_mean_score}\n")
        f.write(f"Final mean score on test data: {final_test_score}\n")


        # Regression Edited KNN
        print("Starting Regression Edited KNN")
        f.write("Starting Regression Edited KNN\n")
        best_edited_regression_params = tune_hyperparameters(EditedKNN, X_train, y_train, edited_regression_param_grid, mode='regression')
        f.write(f"Best hyperparameters for EditedKNN regression: {best_edited_regression_params}\n")
        mean_score, scores = cross_validation(EditedKNN(**best_edited_regression_params), X_train, y_train, k_folds=5, mode='regression', stratify=False)
        f.write(f"Mean cross-validation score on training data for EditedKNN regression: {mean_score}\n")
        final_mean_score, final_test_score = final_evaluation(EditedKNN, X_train, y_train, X_test, y_test, best_edited_regression_params, mode='regression', datasetname=dataset_name)
        f.write(f"Final mean score on cross-validation for EditedKNN regression: {final_mean_score}\n")
        f.write(f"Final mean score on test data for EditedKNN regression: {final_test_score}\n")

        f.write("\n\n\n")

    # Abalone dataset tests (Regression)
    abalone_path = '../Data/abalone/abalone.data'
    abalone_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    abalone_df = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    abalone_df_encoded = encode_categorical_features(abalone_df)
    X_abalone = abalone_df_encoded.drop(columns=['Rings']).values
    y_abalone = abalone_df_encoded['Rings'].values

    X_abalone_normalized = min_max_normalization(X_abalone)
    perform_tests_regression("Abalone", X_abalone_normalized, y_abalone)


    # Breast Cancer dataset Tests (Classification)
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

    X_breast_cancer_normalized = min_max_normalization(X_breast_cancer)
    perform_tests("Breast Cancer", X_breast_cancer_normalized, y_breast_cancer)


    # Car Evaluation dataset Tests (Classification)
    car_path = '../Data/carevaluation/car.data'
    car_columns = [
        'Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class'
    ]
    car_df = pd.read_csv(car_path, header=None, names=car_columns)

    # Separate the target column before encoding, need the class as that is what it's attempting to find
    y_car = car_df['Class'].values
    X_car_df = car_df.drop(columns=['Class'])

    # Encode only the feature columns
    X_car_encoded = encode_categorical_features(X_car_df)
    X_car = X_car_encoded.values

    # Encode the target values,
    y_car, car_label_mapping = encode_labels(y_car)

    X_car_normalized = min_max_normalization(X_car)
    perform_tests("Car", X_car_normalized, y_car)


    # Preprocess Machine dataset (Regression)
    machine_path = '../Data/computerhardware/machine.data'
    machine_columns = [
        'Vendor name', 'Model name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'
    ]
    machine_df = pd.read_csv(machine_path, header=None, names=machine_columns)
    y_machine = machine_df['PRP'].values
    X_machine_df = machine_df.drop(columns=['PRP'])

    # Encode only the feature columns
    X_machine_encoded = encode_categorical_features(X_machine_df)
    X_machine = X_machine_encoded.values

    X_machine_normalized = min_max_normalization(X_machine)
    perform_tests_regression("Machine", X_machine_normalized, y_machine)


    # Preprocess House Votes 84 dataset (Classification)
    house_votes_path = '../Data/congressionalvotingrecords/house-votes-84.data'
    house_votes_columns = [
        'Class name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
        'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 
        'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'
    ]
    house_votes_df = pd.read_csv(house_votes_path, header=None, names=house_votes_columns)

    # Replace missing values
    house_votes_df.replace('?', np.nan, inplace=True)
    house_votes_df.fillna(house_votes_df.mode().iloc[0], inplace=True)

    # Encode the target values
    y_house_votes = house_votes_df['Class name'].values
    y_house_votes, house_votes_label_mapping = encode_labels(y_house_votes)

    # Encode the feature columns
    X_house_votes_df = house_votes_df.drop(columns=['Class name'])
    X_house_votes_encoded = encode_categorical_features(X_house_votes_df)
    X_house_votes = X_house_votes_encoded.values

    X_house_votes_normalized = min_max_normalization(X_house_votes)
    perform_tests("House Votes", X_house_votes_normalized, y_house_votes)


    # Preprocess Forest Fires dataset (Regression)
    forest_fires_path = '../Data/forestfires/forestfires.csv'
    forest_fires_df = pd.read_csv(forest_fires_path)

    forest_fires_df['log_area'] = np.log(forest_fires_df['area'] + 1)

    # Separate features and transformed target
    X_forest_fires_df = forest_fires_df.drop(['area', 'log_area'], axis=1)
    y_forest_fires = forest_fires_df['log_area'].values

    # Encode the feature columns
    X_forest_fires_encoded = encode_categorical_features(X_forest_fires_df)
    X_forest_fires = X_forest_fires_encoded.values

    X_forest_fires_normalized = min_max_normalization(X_forest_fires)
    perform_tests_regression("Forest Fires", X_forest_fires_normalized, y_forest_fires)