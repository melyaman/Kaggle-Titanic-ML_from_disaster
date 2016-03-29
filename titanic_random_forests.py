#!/usr/bin/env python
# -*- coding: utf-8 -*
import pandas

# Reading the file into a dataframe.

titanic = pandas.read_csv("Data/titanic_train.csv")


# Having a look at the dataframe.

## cleaning the training data set.

# filling the empty fields in the column age with the median walue of the column.
print(titanic.head)
print(titanic.describe())
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

# Find all the unique genders -- the column appears to contain only male and female.
print(titanic["Sex"].unique())
# Replace all the occurences of male with the number 0 and female with 1.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
# Find all the unique values for "Embarked".

print(titanic["Embarked"].unique())

titanic["Embarked"]=titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

## Machine Learning part

# Importing the Random forest module. 
# Generate cross validation folds (3) for the titanic dataset.
# importing helper to do cross validation from SKlearn.

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)


alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())


## processing the test set the same way we processed the training set.

titanic_test = pandas.read_csv("Data/titanic_test.csv")

titanic_test = pandas.read_csv("Data/titanic_test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

########################### running alg on test set and generating output ################################







# Create a new dataframe with only the columns Kaggle wants from the dataset.
#submission = pandas.DataFrame({
#        "PassengerId": titanic_test["PassengerId"],
#        "Survived": predictions
#    })
# generate output file to be submitted to kaggle 
#submission.to_csv("kaggle_titanic.csv",index=False)