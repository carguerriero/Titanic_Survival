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
```
## Loading data
We will import our datasets for training and test. Datasets are taken from Kaggle and they are available for download [here](https://www.kaggle.com/c/titanic/data).
```
# Import datasets
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
```
### Take a look at the data!
```
print ("The shape of the train data is (row, column):"+ str(train.shape))
print (train.info())
print ("The shape of the test data is (row, column):"+ str(test.shape))
print (test.info())
```
The training set includes our **target variable** (also known as dependent variable), the passenger survival status, along with other independent features like gender, class, fare, etc.
The test set does not provide passengers survival status. As a matter of fact, *the test set should be used to see how well our model performs on unseen data*, meaning that the  machine learning model have no relation to the test data.
