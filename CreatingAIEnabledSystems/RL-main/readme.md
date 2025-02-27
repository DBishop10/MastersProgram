# Email Marketing System

* Note: As this repo contained files that were already here any not made by me I will only be adding stuff that I worked on for Module 14 assignment.

## Files

### app.py

This file is what runs our flask app so we can query the model

### model.py

This file contains the definition of the model, this is mostly so app.py doesn't contain it and looks cleaner.

### Dockerfile

This is a simple dockerfile to build the docker image

### Q_table.csv

This is the generated Q_table that was created by the model and utilized by the docker app.

### EmailCampaign.ipynb

This is the jupyter notebook that contains all items that was used to create the Q_table, including the requested reasoning for each part of the model selection. 

### data_examination.ipynb

This was only utilized to check for certain data points to test against, nothing in here is useful to look at.

## How to Run

### Docker Container

1. Pull Docker Image from repository
2. Run Docker Image with `docker run -p 8080:8080 dbishop7/705.603:Assignment8_1`
3. Open Postman
4. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
5. Select the `Body` tab and under it the `raw` selector
6. Add Data in this format `{"gender": "F", "type": "C", "age": 21, "tenure": 16}`, select JSON as the type
7. Press `Send`
8. The Models response will be at the bottom of the screen

### Outside Docker Container

3. In the RL-main directory run python3 app.py, this will open the flask app that we have generated
4. Open Postman
5. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
6. Select the `Body` tab and under it the `raw` selector
7. Add Data in this format `{"gender": "F", "type": "C", "age": 21, "tenure": 16}`, select JSON as the type
7b. If you added an entire row from the csv that will also work, above is the required data points
8. Press `Send`
9. The Models response will be at the bottom of the screen