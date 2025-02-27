# Fraud Detection

## Files

### app.py

This file is what runs our flask app so we can query the model

### data_pipeline.py

This python file manages modifying the data for our model to both train on and convert new data to understand

### dataset.py

Contians methods to create the datasets for training, testing, and validating

### Dockerfile

This is a simple dockerfile to build the docker image

### fraud_detector_model.pkl

Is the saved model that I used for the docker image, will be overwritten if Fraud_Detection.train() is used

### fraud_test.json

Test json file to send to the algorithm, should return fraud

### metrics.py

Gets the metrics of the model the is generated

### model.py

This allows you to train, test, and predict with a model. Train will generate a pkl file that is saved and used in the predict method. If train is run once and the pkl is generated it does not need to be run again unless you made changes to the model

### notfraud_test.json

Test json file to send to the algorithm, should return not_fraud

### SystemsPlan.md

Contains the System Plan for Delivery A

### analysis/exploratory_data_analysis.ipynb

Contains all graphs and answers to questions posed for Delivery B

### analysis/model_performance.ipynb

Contains code to generate 3 different types of models as well as my thoughts on best fits for this problem.

## How to Run

### Docker Container

1. Pull Docker Image from repository
2. Run Docker Image with `docker run -p 8080:8080 dbishop7/705.603:Assignment5_1`
3. Open Postman
4. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
5. Select the `Body` tab and under it the `form-data` selector
6. Add a new key called `file` and select `File` from the dropdown
7. Add either `notfraud_test.json` or `fraud_test.json` or a custom file setup like the other two
8. Press `Send`
9. The Models response will be at the bottom of the screen

### Outside Docker Container

1. If fraud_detector_model.pkl does not exist you will need to generate the model, if it does exist skip to step 3
2. Run GenerateModel.py, the only outside file you should need is `Transactions.csv`
3. In the FraudDetection directory run python3 app.py, this will open the flask app that we have generated
4. Open Postman
5. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
6. Select the `Body` tab and under it the `form-data` selector
7. Add a new key called `file` and select `File` from the dropdown
8. Add either `notfraud_test.json` or `fraud_test.json` or a custom file setup like the other two
9. Press `Send`
10. The Models response will be at the bottom of the screen