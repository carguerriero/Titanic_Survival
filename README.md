# Titanic_Survival
This tutorial aims to give an accessible introduction to how to use machine learning techniques for beginners.
The objective is to predict survival on the Titanic and get familiar with ML basics! Therefore, I am going to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

## Problem definition
Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, the scope of our model will be to determine (based on a given test dataset not containing the survival information) if these passengers in the test dataset survived or not.

# 1 - Importing Necessary Libraries and Data
## Loading libraries :books:
We will start our Python script by loading the necessary libraries for our project.
```
#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
```
## Loading data :floppy_disk:	
We will import our datasets for training and test. Datasets are taken from Kaggle and they are available for download [here](https://www.kaggle.com/c/titanic/data).
```
#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
```
### Data Analysis: take a look at the data! :mag_right:
```
#take a look at the training data
train.describe(include="all")
```
The training set includes our **target variable** (also known as dependent variable), the passenger survival status, along with other independent features like gender, class, fare, etc.
The test set does not provide passengers survival status. As a matter of fact, *the test set should be used to see how well our model performs on unseen data*, meaning that the  machine learning model have no relation to the test data.
Having loaded both datasets, we can define **our goal**: we want to find patterns in train.csv that help us predict whether the passengers in test.csv survived.
```
#list of the features within the dataset
print(train.columns)
#check for any other unusable values
print(pd.isnull(train).sum())
```
The Age feature is missing approximately 19.8% of its values. We can guess that Age is important for survival, so we should probably attempt to fill these gaps.
The Cabin feature is missing approximately 77.1% of its values. Since so much of the feature is missing, it would be hard to fill in the missing values. We'll probably drop these values from our dataset.

## Data Viz :bar_chart:
### Feature: Sex
```
#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
```
From the graph, Females have a much higher chance of survival than males. Therefore, Sex feature is essential in our predictions.
### Feature: Pclass
```
#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
```
As predicted, people with higher socioeconomic class had a higher rate of survival.
