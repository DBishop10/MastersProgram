# Fraud Detection

## Files

### app.py

This file is what runs the model itself, it accepts a url or video file input

### data_pipeline.py

This python file manages modifying the data for our model to both train on and convert new data to understand

### dataset.py

Contians methods to create the datasets for training, testing, and validating

### Dockerfile

This is a simple dockerfile to build the docker image

### metrics.py

Gets the metrics of a model

### model.py

This is the backend of the code and contains a test and predict function. Predict is run through app.py to find license plates in the video stream.

### SystemsPlan.md

Contains the System Plan for Delivery A

### analysis/exploratory_data_analysis.ipynb

Contains all answers to questions posed for Delivery B

### analysis/model_performance.ipynb

Contains code to find the metrics of the two models.

## How to Run

### Outside Docker Container
1. If you have a specific stream you want it to link to set an environment variable UDP_URL pointing to it
2. Run `python app.py`
3. The Models response will be in the terminal