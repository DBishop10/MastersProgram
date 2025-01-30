# Beginning Readme, to be updated with final code submission

## Files and what they do

### NullModel.py
This file contains the Null Version of the models, this is requested in part 2 of the assignment

### NonParametric.py
This file contains the KNN and Edited KNN Algoirthm code. I hope to make a few more updates before fully submitting this code. It also contains the code for euclidean_distance, gaussian_kernel, and 2 normalization methods (Not fully tested yet)

### Cross_Validation.py
This file is my way of making repeatable functions for testing as well as performing the cross validation. This houses the tuning_hyperparameters function to ensure I utilize the "best" variables

### Tests.py
This is where I test my 2 KNN models against our 6 datasets, I test both models for regression and classification on each dataset. The information about this is written to Tests.txt in a human readable format.

## Notes

- The code should be runnable simply by calling python Tests.py, after changing database locations as I presumed you didnt want those included. I utilized Python 3.10 during my coding and testing.
- The only packages needed should be numpy and pandas as I utilized some of thier data handling functions
- Some of the data does not get predicted well in the current code, I plan on tuning the model more before final submission
- Let me know if there is too much np or pd in my code, happy to create actual functions, utilizing theres just made life a little easier
- Better comments coming