# Amazon Reviews

## Files

### app.py

This file is what runs our flask app so we can query the model

### data_pipeline.py

This python file manages modifying the data for our model to both train on and convert new data to understand

### Dockerfile

This is a simple dockerfile to build the docker image

### random_forest_model.joblib

Is the saved model that I used for the docker image, may be unavaliable from github repo due to size, but can be generated in `Generate_Model.ipynb`

### metrics.py

Gets the metrics of the model that is generated

### model.py

This allows you to predict with a model

### ModelPlan.md

Contains the Model Plan for First Delivery

### analysis/Data_Analysis.ipynb

Contains all data analysis done on the amazon review dataset for Second Delivery

### Model_Metrics/metrics.ipynb

Contains code on different models and how they performed with my data.

## How to Run

### Docker Container

1. Pull Docker Image from repository
2. Run Docker Image with `docker run -p 8080:8080 dbishop7/705.603:Assignment7_1`
3. Open Postman
4. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
5. Select the `Body` tab and under it the `raw` selector
6. Add Data in this format `{"review_title": "Good Movie", "text": "I thought it was good!", "verified_purchase": true, "helpful_vote": "1"}`, select JSON as the type
6b. If you added an entire row from the csv that will also work, above is the required data points
7. Press `Send`
8. The Models response will be at the bottom of the screen

### Outside Docker Container (Can Only Be Done if you Generate A Model)

1. If fraud_detector_model.pkl does not exist you will need to generate the model, if it does exist skip to step 3
2. Open Generate_Model.ipynb, the only outside file you should need is `amazon_movie_reviews.csv`, run all items inside ipynb file
3. In the FraudDetection directory run python3 app.py, this will open the flask app that we have generated
4. Open Postman
5. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
6. Select the `Body` tab and under it the `raw` selector
7. Add Data in this format `{"review_title": "Good Movie", "text": "I thought it was good!", "verified_purchase": true, "helpful_vote": "1"}`, select JSON as the type
7b. If you added an entire row from the csv that will also work, above is the required data points
8. Press `Send`
9. The Models response will be at the bottom of the screen