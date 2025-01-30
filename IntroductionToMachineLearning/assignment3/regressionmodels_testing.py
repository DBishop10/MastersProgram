from utility import *
from regressionmodels import *
from performance_graphs import *

def cross_val_metrics_logistic_linear(model, X, y, cv=5, repeats=2, task='classification'):
    """
    Performs the Cross Validation on the Logistic or Linear Regression Models

    Parameters:
    model (Logistic or Regression): The model that was trained on the dataset
    X (numpy.ndarray): All X Values of the dataset
    y (numpy.ndarray): All true y values of the dataset
    cv (int): How many Cross Validations you want to do
    repeats (int): How many times you want to repeat the model performance
    task (string): Whether it is a classification or regression dataset
    """
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mse': [],
        'r_squared': []
    }
    for _ in range(repeats):
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        fold_size = len(y) // cv
        for fold in range(cv):
            start = fold * fold_size
            end = start + fold_size if fold != cv - 1 else len(y)
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)
            X_test = X[start:end]
            y_test = y[start:end]

            model.fit(X_train, y_train)

            if task == 'classification':
                score = evaluate_classification(model, X_test, y_test)
            else:
                score = evaluate_regression(model, X_test, y_test)
            for key, value in score.items():
                if key in scores:
                    scores[key].append(value)
    
        mean_scores = {key: np.mean(values) for key, values in scores.items() if values}
        std_scores = {key: np.std(values) for key, values in scores.items() if values}

    return mean_scores, std_scores

def load_datasets():
    """
    Loads each dataset into a dict for easier training

    Return:
    datasets (dict): Contains all the loaded and pipelined datasets
    """
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
    abalone_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
    ]
    abalone = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    abalone['Sex'] = abalone['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    y_abalone = abalone['Rings'].values.reshape(-1, 1)
    X_abalone = abalone.drop(columns=['Rings']).values
    datasets['Abalone'] = (X_abalone, y_abalone, 'regression')

    # Car Evaluation dataset (Classification)
    car_path = '../Data/carevaluation/car.data'
    car_columns = [
        'Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class'
    ]
    car_df = pd.read_csv(car_path, header=None, names=car_columns)
    y_car = car_df['Class'].values
    X_car_df = car_df.drop(columns=['Class'])
    X_car_encoded = encode_categorical_features(X_car_df)
    X_car = X_car_encoded.values
    y_car, car_label_mapping = encode_labels(y_car)
    datasets['Car Evaluation'] = (X_car, y_car, 'classification')

    # Machine dataset (Regression)
    machine_path = '../Data/computerhardware/machine.data'
    machine_columns = [
        'Vendor name', 'Model name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'
    ]
    machine_df = pd.read_csv(machine_path, header=None, names=machine_columns)
    y_machine = machine_df['PRP'].values.reshape(-1, 1)
    X_machine_df = machine_df.drop(columns=['PRP', 'Vendor name', 'Model name'])
    X_machine_encoded = pd.get_dummies(X_machine_df)  
    X_machine = min_max_normalization(X_machine_encoded.values)  
    y_machine = min_max_normalization(y_machine)  
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

    X_house_votes_df = house_votes_df.drop(columns=['Class name'])
    X_house_votes_encoded = encode_categorical_features(X_house_votes_df)
    X_house_votes = X_house_votes_encoded.values

    datasets['House Votes 84'] = (X_house_votes, y_house_votes, 'classification')

    # Forest Fires dataset (Regression)
    forest_fires_path = '../Data/forestfires/forestfires.csv'
    forest_fires_df = pd.read_csv(forest_fires_path)
    forest_fires_df['log_area'] = np.log(forest_fires_df['area'] + 1)
    X_forest_fires = forest_fires_df.drop(['area', 'log_area'], axis=1)
    X_forest_fires_encoded = encode_categorical_features(X_forest_fires)
    X_forest_fires = X_forest_fires_encoded.values
    y_forest_fires = forest_fires_df['log_area'].values.reshape(-1, 1)
    X_forest_fires = min_max_normalization(X_forest_fires)
    y_forest_fires = min_max_normalization(y_forest_fires)
    datasets['Forest Fires'] = (X_forest_fires, y_forest_fires, 'regression')
    
    return datasets

if __name__ == "__main__":
    datasets = load_datasets()
    file = open("output/linearlog.txt", "a")
    for dataset_name, (X, y, task) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=(task == 'classification'))

        if task == 'classification':
            model = LogisticRegression(learning_rate=0.0001, num_iterations=10000)
        else:
            model = LinearRegression(learning_rate=0.01, num_iterations=10000)

        mean_score, std_score = cross_val_metrics_logistic_linear(model, X_train, y_train, cv=5, repeats=2, task=task)
        file.write(f'{model.__class__.__name__} ({dataset_name}) - Mean Score: {mean_score}, Std: {std_score} \n')
        #Generate Graphs
        if task == 'classification':
            plot_confusion_matrix(model, X_test, y_test, f"graphs/{dataset_name}/logistic")
        else:
            plot_regression_metrics(model, X_test, y_test, f"graphs/{dataset_name}/linear")
    file.close()