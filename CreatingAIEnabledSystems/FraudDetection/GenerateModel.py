from model import Fraud_Detector_Model
import pandas as pd
from data_pipeline import ETL_Pipeline
from dataset import Fraud_Dataset
from metrics import Metrics 

pipeline = ETL_Pipeline()
metrics = Metrics()
fraud_detector = Fraud_Detector_Model()

raw_data = pipeline.extract('transactions.csv')
transformed_data = pipeline.transform(raw_data)
pipeline.load(transformed_data, 'transformed_transactions.csv')

fraud_dataset = Fraud_Dataset(transformed_data)

X, y = transformed_data.drop('is_fraud', axis=1), transformed_data['is_fraud']
fraud_detector.train(X, y)