# Motivation

## Why are we solving this problem?

We are addressing this issue to enhance the security and trustworthyness of financial services for out hypothetical job. The rising number of fraudulent transactions affects the companies profitability through both losses as well as a damaged reputation. Solving this issue will minimize company losses as well as boosting customer confidence.

### Problem Statement

There needs to be an improvement on the existing machine learning model for detecting fraud as it currently has an issue with precision and recall scores, which are approximately 40% and 70% respectively.

### Value Proposition

My new ML algorithm will significantly reduce financial losses due to fraud and give customers more confidence in our work.

## Does an ML solution “fit” this problem?

An ML solution does fit this problem as we can teach if from historical transaction data to help it learn to predict and identify patterns of fraud. This approach would also be scalable meaning it could continuously evolve with new data.

## Are we able to tolerate mistakes and some degree of risk?

A certain degree of risk is tolerable. Due to the nature of fraud detection false postitives and false negatives are expected. While we want to minimize occurances of this, no system can be 100% accurate.

## Is this solution feasible? 

As we already know that one ML solution was already created, developing a more accurate and possibly more efficient model is feasible.

# Requirements

## Scope:

### What are out goals?

We have two goals, one is to develope a new ML model that improves upon the current model's precision and recall scores. The second is to ensure that the new model can be integrated with existing systems with minimal issues.

### What are the success criteria?

The main success criteria is to have precision and recall scores higher than the current 40% and 70%. 

## Requirements:

### What are our (system) Assumptions?

We will be assuming that historical transaction data is sufficiently comprehensive regrading fraudulence. We will also assume that the patterns of fraud in the past will correlate to detecting future fraud.

### What are our (system) Requirements?

The system must process transactions in near-real-time to prevent fraud effectively. It must handle the data volume and velocity of the banks transaction processing system. And finally the model should be explaniable to some degree as customers may request a review on a false positive.

## Risk & Uncertanties:

### What are the possible harms?

False positives could lead to customer dissatisfaction. False negatives could result in financial loss to the company.

### What are the causes of mistakes?

Incomplete or biased traning data can cause issues. Overfitting or underfitting the model can cause major issues in false positives or negatives.

# Implementation

## DEVELOPMENT:

### Methodology

Utilizing ML techniques such as deep learning or ensemble methods will help detect fraud patterns.
 
### High-level System Design

Integrate the new ML algorithm into the transaction processing pipeline.

### Development Workflow

First step will be data preprocessing and exploration to get a better grasp on features and patterns. Then I will utilize that data in model traning, validation, and testing using the historical data. Finally post that I would work on continuing to evaulate the tuning of the model based on new data and feedback.

### ETL Pipeline Design

The ETL pipeline is designed to process transaction data to facilitate fraud detection modeling. The pipeline consists of the following stages:

- **Extract**: The `extract` method retrieves data from a provided CSV file. This stage is responsible for the initial data ingestion.

- **Transform**: The `transform` method applies a series of data cleaning and preparation steps. These transformations are based on insights from the exploratory data analysis. For example, the transformation step may include parsing dates and times, encoding categorical variables, deriving new features (such as the hour of the day from transaction timestamps), and filtering or imputing missing values.

- **Load**: The `load` method saves the transformed data into a specified CSV file, making it ready for use in fraud detection models. This stage outputs a clean, pre-processed dataset that serves as the foundation for predictive analytics.

### Dataset Partitioning Strategy

To evaluate our fraud detection model effectively, we will implement a robust dataset partitioning strategy using our `Fraud_Dataset` class. This strategy encompasses the following:

- **Initial Split**: We first partition our dataset into training (60%) and testing (20%) datasets, along with a separate validation set (20%) to fine-tune our model's hyperparameters.

- **k-Fold Cross-Validation**: To ensure our model's performance is consistent across different subsets of data, we will utilize k-fold cross-validation with `k` set to 5 by default. This approach helps us to mitigate overfitting and provides a more reliable performance estimate.

- **Training Set**: The training set is used to fit our model and learn the data patterns associated with fraudulent transactions.

- **Validation Set**: The validation set acts as a proxy for the test set and is used to adjust the model's hyperparameters to improve performance.

- **Testing Set**: The testing set is held out from all model training and validation processes. It provides an unbiased evaluation of the final model fit on the training dataset.

- **Cross-Validation Generator**: The `get_kfold_datasets()` method yields pairs of training and validation sets for each fold, allowing us to systematically train and validate our model across different data segments.

### Metrics Pipeline

I have established a metrics pipeline to evaluate the performance of my fraud detection model, this has been implemented in the `metrics.py` module. This will help us understand how well our model meets its objectives, particularly in improving upon the existing model's precision and recall scores.

## Metrics Class

The `Metrics` class created within `metrics.py` module is designed to calculate key performance metrics including, precision, recall, sensitivy, specificity, ROC-AUC, and Precision-Recall Curves.

- **Precision**: This will measure the accuracy of positive predictions. In the context of fraud detection, high precision recall can help reduce false positives. This is critical for our problem as having too many false positives will damage reputation with customers.

- **Recall**: Assesses the model's ability to detect all actual fraud cases. High recall rate is vital to ensure that fraudulent transactions are identified. If we do not have this metric or it is low then we would be unable to accurately assess the model and make adjustments as needed.

- **Sensitivity**: This is very similar to recall in that it assesses the models ability to detect actual fraud cases. This furthers the idea that we need these metrics to get an actual overview of how our model is doing.

- **Specificity**: Measures the model's ability to identify legitimate transactions correctly. Much like with precision this is important as making too many false positives will end up upsetting customers and possibly losing out on more buisiness.

- **ROC-AUC**: This evaluates the model's effectiveness in seperatign the fraudulent transactions from non-fraudulent ones. This is very important as it gives a good overview of where the model is at in identifying fraud cases without too many false positives.

- **Precision-Recall Curve** Directly assesses the trade off between precision and recall. For this project this would be an incredibly important metric as low recall will lead to missing fraudulent transactions. Utilizing this as well helps us adjust the model as the data is so one sided for fraudulent data to non-fraudulent data.

## POLICY:

### Human-Machine Interfaces

I will be utilizing Postman to input .json files for the model to read.

### Regulations

We would need to follow relevant financial regulations and data protection laws. 

## OPERATIONS:

### Continuous Deployment

Utilizing A/B testing to compare the performance of new models against the older versions.

### Post-deployment Monitoring

A good item for post deployment would be an alert system for anomalies or deteriorations in model performance.

### Maintenance

Regular updates to the model based on any new data or emerging patterns.

### Quality Assurance

We would need to perform rigorous testing of the model before deployment. Benchmarks should also be established for model performance and mechanisms for continuous improvement.