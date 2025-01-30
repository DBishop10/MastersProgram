## Simple Overview

### Model files
There are three model files, regressionmodels.py, feedforward_nn.py, and autoencoder.py. These files contain the code to train and test with the model

### Utility
utility.py contains all items used for determining how the models are performing including the different scores. This file also contains the methods to split data and a few other useful methods.

### Testing Files
As each of these models wanted data in slightly different ways, unlike in previous assignments, I have split out the testing file into 3 files, one for each model type. This ensures I can train the models as needed and get much better data for each one

### videotest.py
This file contains all the methods created to demonstrate each requirement for the demo video.

### graphs folder
This folder contains all generated graphs of each model on each dataset. Note: The mse_plot and r2_plot are exactly the same except they display the corresponding metric at the top of the graph.

### output folder
This folder contains three text files, these files are the model performances during their training that I performed. The data contained here is reflected in the paper.