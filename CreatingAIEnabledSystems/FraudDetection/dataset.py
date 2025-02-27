from sklearn.model_selection import train_test_split, KFold
import pandas as pd

class Fraud_Dataset:
    """
    A class to handle the partitioning of a dataset into training, testing, and validation sets.
    
    Attributes
    ----------
    data : DataFrame
        The entire dataset that will be split.
    k_folds : int
        The number of folds to use for k-fold cross-validation.

    Methods
    -------
    get_training_dataset():
        Returns the training dataset.
        
    get_testing_dataset():
        Returns the testing dataset.
        
    get_validation_dataset():
        Returns the validation dataset if k-fold cross-validation is not used.
        
    get_kfold_datasets():
        Yields k-fold training and validation datasets for use in cross-validation.
    """
    
    def __init__(self, data, k_folds=5):
        """
        Constructs all the necessary attributes for the Fraud_Dataset object.

        Parameters
        ----------
        data : DataFrame
            The entire dataset to be split.
        k_folds : int, optional
            The number of folds for k-fold cross-validation (default is 5).
        """
        self.data = data
        self.k_folds = k_folds
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        
        # Split the data initially to avoid multiple splits in each function
        self.initial_split()

    def initial_split(self):
        """
        Splits the data into training and testing datasets.
        """
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        # Further split training data to create a validation set if not using k-fold
        self.train_data, self.validation_data = train_test_split(self.train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    def get_training_dataset(self):
        """
        Returns the training dataset.

        Returns
        -------
        train_data : DataFrame
            The training dataset.
        """
        return self.train_data

    def get_testing_dataset(self):
        """
        Returns the testing dataset.

        Returns
        -------
        test_data : DataFrame
            The testing dataset.
        """
        return self.test_data

    def get_validation_dataset(self):
        """
        Returns the validation dataset.

        Returns
        -------
        validation_data : DataFrame
            The validation dataset.
        """
        return self.validation_data

    def get_kfold_datasets(self):
        """
        Generator that yields training and validation datasets for each fold of the k-fold cross-validation.

        Yields
        ------
        fold_train_data : DataFrame
            The training dataset for the current fold.
        fold_val_data : DataFrame
            The validation dataset for the current fold.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(self.train_data):
            fold_train_data = self.train_data.iloc[train_index]
            fold_val_data = self.train_data.iloc[val_index]
            yield fold_train_data, fold_val_data