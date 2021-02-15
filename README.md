# Titanic_Survival
This tutorial aims to give an accessible introduction to how to use machine learning techniques for beginners.
The objective is to predict survival on the Titanic and get familiar with ML basics! Therefore, I am going to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

## Problem definition
Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, the scope of our model will be to determine (based on a given test dataset not containing the survival information) if these passengers in the test dataset survived or not.

# 1 - Importing Necessary Libraries and Data
## Loading libraries
We will start our Python script by loading the necessary libraries for our project.
```
# Import
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

%matplotlib inline

import warnings ## importing warnings library. 
warnings.filterwarnings('ignore') ## Ignore warning
```
And we will importing our datasets for training and test.
```
# Import datasets
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
```
