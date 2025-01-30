## Simple Overview

### trackandcar.py
This file contains the base for the track and car utilized by all models

### Model Files

#### qlearningSARSA.py
Contains code for the Q-learning and SARSA models. Also contains some testing functions at the bottom (not robust just enough to ensure they work on each track)

#### valueiteration.py
Contains code for the Valueiteration model

### Testing Files

#### tests.py
This File has all items required to test each model on each course in a round robin type of way. It is important to note that SARSA has an issue with one of the rewards, this can be seen in the qlearningSARSA.py reward function, nothing is required to be changed, just good to note

#### video_tests.py
This file contains all code for the tests shown in the attached video.

### Output

#### tests.txt
This file houses all data collected accross 10 runs of each model on each track and reset condition. tests.py will append to the end of the file instead of erasing everything. It currently contains all data found in the assignment paper.

### How to Run
There are a few ways to run these. It is important to note the only package you will need to install is numpy. You will also need to either set a Data directory one level up that contains a folder called track with all of the tracks in it, or change the path that the files are looking for manually.

#### Track Alone
If you just want to see the track work you can look at the bottom of the trackandcar.py file where you will see a few tests I ran personally. You would simply need to run `python trackandcar.py`with any track you would like.

#### Q-learning and SARSA
Once again at the bottom in the main function you will see where to change the track and how to run both Q-learning and SARSA models. Run `python qlearningSARSA.py`

#### Value Iteration
Much like the other files there are already some test items in the bottom main function.  Run `python valueiteration.py` to see the results

#### Everything at once
The tests.py is setup to run all of the models on all of the tracks as well as output them to tests.txt log file. This is a simpler way to run all of the models. It is important to note that some of the tracks the models train much slower on, especially R-track, lots of time spent training on R-track... running `python tests.py` will run this file.