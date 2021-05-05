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
# 2 - Data Analysis 
## Take a look at the data! :mag_right:
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
### Feature: Age
```
#sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
```
Babies are more likely to survive than any other age group.

## Cleaning Data :broom:
We are now ready to clean the data and drop irrelevant features.
We will start off by dropping the Cabin and Ticket feature since not a lot more useful information can be extracted from them.
```
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
```
Consequently, we need to fill the missing values, according to the distribution of the single feature.
We will start with the Embark feature.
```
print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)
```
The majority of people embarked in Southampton (S). We can go ahead and fill in the missing values with the value 'S'.
```
#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})
```
What about Age? A high percentage of values are missing, so we need to fill this gap. Usually, the most common solution would be to replace the missing values with the median for everyone of our passengers in the dataset, but let's try to predict the age for categoryies instead, as differente characteristics would lead to different ages.
```
#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```
Next, we'll try to predict the missing Age values from the most common age for their Title.
```
#fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
```
