## Files and what they do

### DecisionTree.py
Contains the Code for the Decision Trees, it also includes the Node class that allows for easy debugging as each node has specific variables which can be called.

### GenerateGraphs.py
Contains the code to generate the graphs for pruned and non pruned tree metrics, not necessary per assignment but made my life a lot easier.

### Tests.py
This is where I test the decision tree regressors and Classifiers, it generates a tree for each dataset and then trains it, gets the scores through 5x2 cross validation, prunes it, and gets the final scores again through cross validation on the pruned trees.

### Video_Tests.py
This is similar to the Tests file, but contains all necessary code to demo each feature requested for the video.

## Notes

- The code should be runnable simply by calling python Tests.py, please ensure correct database locations as it will not function without them in a reachable spot. This will generate graphs in the graphs folder (graphs contianed in paper are included in this path already) 
- I utilized Python 3.10 during my coding and testing.
- The only packages needed should be numpy and pandas as I utilized some of thier data handling functions
