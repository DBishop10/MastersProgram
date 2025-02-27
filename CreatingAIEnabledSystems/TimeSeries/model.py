from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from joblib import load
import pandas as pd

class Model:
    """
    A class used to represent an ETL pipeline for transaction data.

    ...

    Attributes
    ----------
    model: The model to use
    scalerx: The scalerx to use
    scalery: The scalery to use

    Methods
    -------
    change_model(model_path):
        Change what model you want to use
        
    change_scalerx(scalerPath):
        Change what scalerx you want to use
        
    change_scalery(scalerPath):
        Change what scalery you want to use
    """
    
    def __init__(self):
        self.model = load_model('finalDeepModel')
        self.scaler_x = load('finalDeepModel/scaler_x.joblib')
        self.scaler_y = load('finalDeepModel/scaler_y.joblib')
    
    def change_model(self, model_path):
        self.model = load_model(model_path)
        
    def change_scalerx(self, scalerPath):
        self.scaler_x = load(scalerPath)
        
    def change_scalery(self, scalerPath):
        self.scaler_y = load(scalerPath)
        
    def prediction(self, input):
        input_date = pd.to_datetime(input)

        sequence_length = 12

        sequence_dates = [(input_date - pd.Timedelta(days=x)).strftime('%m/%d') for x in range(sequence_length)]

        historical_df = pd.read_csv('average_daily_data.csv')

        forecast_input_df = historical_df[historical_df['mm_dd'].isin(sequence_dates)]

        historical_df.drop('mm_dd', axis=1, inplace=True)
        forecast_input_df.drop('mm_dd', axis=1, inplace=True)
        historical_df.drop('total_daily_transactions', axis=1, inplace=True)
        forecast_input_df.drop('total_daily_transactions', axis=1, inplace=True)
        historical_df.drop('total_daily_fraud_transactions', axis=1, inplace=True)
        forecast_input_df.drop('total_daily_fraud_transactions', axis=1, inplace=True)

        forecast_input_scaled = self.scaler_x.transform(forecast_input_df)

        forecast_input_reshaped = forecast_input_scaled.reshape((forecast_input_scaled.shape[0], 1, forecast_input_scaled.shape[1]))

        forecast_input_values = forecast_input_reshaped.astype('float32')


        predicted_outcomes = self.model.predict(forecast_input_values)

        predicted_outcomes_rescaled = self.scaler_y.inverse_transform(predicted_outcomes)

        return predicted_outcomes_rescaled[0, 0], predicted_outcomes_rescaled[0, 1]