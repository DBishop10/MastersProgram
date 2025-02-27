import os
import random
from sklearn.model_selection import KFold

class Object_Detection_Dataset:
    def __init__(self, dataset_directory, seed=42):
        """
        Initializes the dataset object.
        Parameters:
        dataset_directory: Path to the directory containing the dataset.
        seed: Random seed for reproducibility, default 42
        """
        self.dataset_directory = dataset_directory
        self.seed = seed
        self.dataset = self._load_dataset()
        random.seed(seed)

    def _load_dataset(self):
        """
        Loads the dataset from the directory specified during initialization.

        Return:
        List of file paths for the images.
        """
        file_paths = [os.path.join(self.dataset_directory, f) 
                      for f in os.listdir(self.dataset_directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return file_paths

    def _split_dataset(self, validation_split=0.2, test_split=0.2):
        """
        Splits the dataset into training, validation, and testing sets.
        Parameters:
        validation_split: Proportion of the dataset to include in the validation split, default 0.2
        test_split: Proportion of the dataset to include in the test split, default 0.2
        Return: 
        Three lists containing the file paths for the training, validation, and test sets.
        """
        total_size = len(self.dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        test_size = int(total_size * test_split)
        validation_size = int(total_size * validation_split)

        test_indices = indices[:test_size]
        validation_indices = indices[test_size:test_size + validation_size]
        train_indices = indices[test_size + validation_size:]

        train_dataset = [self.dataset[i] for i in train_indices]
        validation_dataset = [self.dataset[i] for i in validation_indices]
        test_dataset = [self.dataset[i] for i in test_indices]

        return train_dataset, validation_dataset, test_dataset

    def get_training_dataset(self):
        """
        Returns the training dataset.

        Return:
        List of file paths for the training images.
        """
        train_dataset, _, _ = self._split_dataset()
        return train_dataset

    def get_validation_dataset(self):
        """
        Returns the validation dataset.

        Return:
        List of file paths for the validation images.
        """
        _, validation_dataset, _ = self._split_dataset()
        return validation_dataset

    def get_testing_dataset(self):
        """
        Returns the testing dataset.

        Return: 
        List of file paths for the testing images.
        """
        _, _, test_dataset = self._split_dataset()
        return test_dataset

    def k_fold_cross_validation(self, k=5):
        """
        Performs k-fold cross-validation on the dataset.
        Parameter:
        k: Number of folds.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        for train_indices, val_indices in kf.split(self.dataset):
            yield [self.dataset[i] for i in train_indices], [self.dataset[i] for i in val_indices]