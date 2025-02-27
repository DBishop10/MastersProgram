import pandas as pd
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from data_pipeline import ETL_Pipeline


class Fraud_Detector_Model:
    """
    A class for constructing a fraud detection model using RandomForestClassifier.

    ...

    Attributes
    ----------
    model : RandomForestClassifier
        The Random Forest classifier used for fraud detection.
    encoders : dict
        A dictionary containing LabelEncoders for categorical variables.

    Methods
    -------
    train(X_train, y_train):
        Trains the RandomForestClassifier on the provided training data.

    predict(X):
        Predicts the class (Fraud or Not Fraud) of the given input data.
    """

    def __init__(self):
        """
        Initializes the Fraud_Detector_Model with a RandomForestClassifier.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoders = {}
        self.pipeline = ETL_Pipeline()
    
    def train(self, X_train, y_train):
        """
        Trains the RandomForestClassifier on the provided training data.

        Parameters
        ----------
        X_train : DataFrame
            The training input samples.

        y_train : Series
            The target labels for training.
        """
        self.model.fit(X_train, y_train)
        # Save the trained model
        joblib.dump(self.model, 'fraud_detector_model.pkl')
    
    
    def test(self, X_test, y_test):
        """
        Tests the trained RandomForestClassifier on the provided test data and evaluates using the Metrics class.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.

        y_test : array-like of shape (n_samples,)
            The true labels for X_test.

        Returns
        -------
        report : str
            Text report showing the main classification metrics.
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Assuming binary classification

        # Evaluate metrics
        metrics = Metrics()
        metrics.generate_report(y_test, y_pred, y_pred_proba)

        # Additionally, you can return the classification report
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        return report
    
    def predict(self, json_input):
        """
        Predicts the class (Fraud or Not Fraud) of the given input data.

        Parameters
        ----------
        json_input : str or dict
            JSON input containing the transaction data. This could be a path to a JSON file 
            or a JSON string/dict

        Returns
        -------
        str
            The predicted class: 'Not Fraud' or 'Fraud'.
        """
        
        # Check if json_input is a string, if so, treat it as a file path
        if isinstance(json_input, str):
            with open(json_input, 'r') as file:
                data_dict = json.load(file)
        else:
            # Otherwise, treat it as a dictionary
            data_dict = json_input
        
        # Convert the JSON/dict to a pandas DataFrame
        data_df = pd.DataFrame([data_dict])
        
        
        transformed_X = self.pipeline.transform(data_df)
        # Load the trained model
        self.model = joblib.load('fraud_detector_model.pkl')
        
        # Obtain predicted probabilities for the positive class
        y_pred_proba = self.model.predict_proba(transformed_X)[:, 1]  # Assuming binary classification

        # Apply a custom threshold
        prediction = (y_pred_proba >= 0.85).astype(int)
        
        # Return the predicted class as a string
        return "Fraud" if prediction[0] == 1 else "Not Fraud"