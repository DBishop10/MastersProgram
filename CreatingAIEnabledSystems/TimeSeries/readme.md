# Fraud Detection

## Files

### analysis/exploratory_data_analysis.ipynb

Contains graphs and thoughts for new data for this project

### Assignment/Part1Models
Contains .ipynb files for the creation of both the traditional and Deep model. It also has an ipynb file for the conclusions of each of those models

### Assignment/Part2Models
Contains .ipynb files for the creation of both the traditional and Deep model. It also has an ipynb file for the conclusions of each of those models

### Assignment/Part3Models
Contains .ipynb files for the creation of both the traditional and Deep model. It also has an ipynb file for the conclusions of each of those models

### Assignment/Part4Models
Contains .ipynb files for the creation of both the traditional and Deep model. It also has an ipynb file for the conclusions of each of those models

### Assignment/readme.md
This readme contained the assignment information

### data

contains all data for above notebooks as well as fraud data of 5 years

### finalDeepModel
This folder contians the saved deep model used by app.py (it was made in Assignments/Part4Models/Part4DeepModel.ipynb)

### app.py

This file is what runs our flask app so we can query the model

### average_daily_data.csv
This csv contains all 5 years data averaged together to utilize in variable forecasting

### data_pipeline.py

This python file manages modifying the data for our model to train on

### GetAverageData.ipynb
Contains the code on how I created the average_daily_data.csv

### model.py

This is a file to call to load the model, mostly used for our app.py

### p1dm.csv
contains the forecasted values that the part 1 deep model forecasted

### p1tm.csv
contains the forecasted values that the part 1 deep traditional forecasted

### p3dm.csv
contains the forecasted values that the part 3 deep model forecasted

### p3dm.csv
contains the forecasted values that the part 3 traditional model forecasted

### TimeSeriesDataPrep.ipynb:

Notebook that explores pandas time methods.

### TimeSeriesTraditionalModels.ipynb: 

Notebook to explore tranditional time series models.

### TimeSeriesDeepLearningModels.ipynb:

Notebook to explore neural network time series models.

### TimeSeriesProphetModel.ipynb: 

Notebook to explore the prophet time series model

## How to Run

1. Run app.py
2. Open Postman
3. Open a new tab and set the request to `POST`, add url like `http://127.0.0.1:8080/predict`
4. Select the `Headers` tab
5. Set the Key to Content-Type and the Value to application/json
6. Select the `Body` tab and select the `raw` selector
7. Add this json format to the space {"date": <Date to Forecast>}
8. Note: Date to Forecast should be in format YYYY-MM-DD
9. Press `Send`
10. The Models response will be at the bottom of the screen