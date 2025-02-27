import pandas as pd
import os
from data_pipeline import Pipeline
from joblib import load

class Amazon_Model:
    """
    A class for constructing a fraud detection model using RandomForestClassifier.

    ...

    Attributes
    ----------
    model : RandomForestRegressor
        The RandomForestRegressor used for polarity
    vectorizor : vectorizor
        Used to vectorize the data
    umap: umap
        Umap for the data
    pipeline: Pipeline
        The data pipeline to feed data through

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
        self.model = load('random_forest_model.joblib')
        self.vectorizor = load('tfidf_vectorizer.joblib')
        self.pipeline = Pipeline()
    
    def categorize_sentiment(self, polarity):
        if polarity < -0.8:
            return "Extremely Negative"
        elif -0.8 <= polarity < -0.5:
            return "Moderately Negative"
        elif -0.5 <= polarity < -0.2:
            return "Slightly Negative"
        elif -0.2 <= polarity < 0.2:
            return "Neutral"
        elif 0.2 <= polarity < 0.5:
            return "Slightly Positive"
        elif 0.5 <= polarity < 0.8:
            return "Moderately Positive"
        else:
            return "Extremely Positive"
    
    def predict(self, json_input):
        """
        Predicts the class of the given input data.

        Parameters
        ----------
        json_input : str or dict
            JSON input containing the transaction data. This could be a path to a JSON file 
            or a JSON string/dict

        Returns
        -------
        str
            The catagorized and raw prediction.
        """
        data_dict = json_input
        
        # Convert the JSON/dict to a pandas DataFrame
        data_df = pd.DataFrame([data_dict])
                
        self.pipeline.vectorizer = self.vectorizor
        # self.pipeline.umap_model = self.umap
         
        transformed_X = self.pipeline.transform(data_df)
        
        transformed_X.drop(['polarity'], axis=1, inplace=True)
        
        # Obtain predicted probabilities for the positive class
        y_pred_proba = self.model.predict(transformed_X)  # Assuming binary classification

        catagorized = self.categorize_sentiment(y_pred_proba);
        
        # Return the predicted class as a string
        return catagorized, y_pred_proba