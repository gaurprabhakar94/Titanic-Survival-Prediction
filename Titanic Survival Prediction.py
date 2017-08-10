# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 01:49:28 2017

@author: PG
"""

import sklearn
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
os.chdir("C:\Users\PG\Desktop\kaggle")
from sklearn.tree import DecisionTreeClassifier
pd.options.display.max_rows = 100

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

#Reading the training dataset
train = pd.read_csv("train.csv", sep = ',', header = 0)
test = pd.read_csv("test.csv", sep=',', header = 0)


#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

#Checking for data types and number of Nan Values
print(train.info())
print("\n\n")
print(train.shape[0]- train.count(0))
print("\n\n")

print(test.info())
print("\n\n")
print(test.shape[0]- test.count(0))
print("\n\n")


#Understanding the numerical data of the training dataset
print(train.describe())
print("\n\n")
print(test.describe())
print("\n\n")

#Understanding the categorical data of the training dataset
print(train.describe(include=['O']))
print("\n\n")
print(test.describe(include=['O']))
print("\n\n")


# Passengers that survived vs passengers that passed away
print(train.Survived.value_counts())

# As proportions
print(train.Survived.value_counts(normalize = True))

# Males that survived vs males that passed away
print(train['Survived'][train['Sex'] == 'male'].value_counts())

# Normalized male survival
print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize =True))

# Females that survived vs Females that passed away
print(train['Survived'][train['Sex'] == 'female'].value_counts())

# Normalized female survival
print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize =True))


labels = train.Survived
train.drop('Survived', 1, inplace =True)
train.Age = train.Age.fillna(train.Age.median())
test.Age = test.Age.fillna(test.Age.median())

dfeng = train.append(test)
dfeng.reset_index(inplace = True)
dfeng.drop('index', 1, inplace=True)


# Convert the male and female groups to integer form
dfeng["Sex"][dfeng["Sex"] == "male"] = 0
dfeng["Sex"][dfeng["Sex"] == "female"] = 1


# Create the column Child and assign to 'NaN'
dfeng["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older but younger than 58 and
# 2 to those who are over 58. Print the new column.
dfeng["Child"][dfeng['Age']<18] = 1
dfeng["Child"][(dfeng["Age"] >=18)] = 0


# Impute the Embarked variable
dfeng["Embarked"] = dfeng.Embarked.fillna("S")

# we extract the title from each name
dfeng['Title'] = dfeng['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
# a map of more aggregated titles
Title_Dictionary = {
                        "Capt":       "Officer", 
                        "Col":        "Officer", 
                        "Major":      "Officer", 
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty", 
                        "Sir" :       "Royalty", 
                        "Dr":         "Officer", 
                        "Rev":        "Officer", 
                        "the Countess":"Royalty",
                        "Dona":       "Royalty", 
                        "Mme":        "Mrs",     
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
# we map each title
dfeng['Title'] = dfeng.Title.map(Title_Dictionary)
dfeng = pd.get_dummies(dfeng, columns = ["Title"])





# Convert the Embarked classes to integer form
dfeng["Embarked"][dfeng["Embarked"] == "S"] = 0
dfeng["Embarked"][dfeng["Embarked"] == "C"] = 1
dfeng["Embarked"][dfeng["Embarked"] == "Q"] = 2

# there's one missing fare value - replacing it with the mean.
dfeng.head(891).Fare.fillna(dfeng.head(891).Fare.mean(), inplace=True)
dfeng.iloc[891:].Fare.fillna(dfeng.iloc[891:].Fare.mean(), inplace=True)




#Computing the family size
dfeng["FamilySize"] = dfeng.SibSp + dfeng.Parch + 1


dfeng.drop(["PassengerId", "Cabin", "Name", "Ticket", "Fare", 'SibSp', "Parch", "Age"], axis=1, inplace=True )


print(dfeng.describe())

print(dfeng.head())



# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = dfeng.iloc[891:].values
train_features = dfeng.head(891)


# Make your prediction using the test set
clf = tree.DecisionTreeClassifier()
clf.fit(train_features, labels)
prediction = clf.predict(test_features)
clf = prediction


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)

solution = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
print(solution)

# Check that your data frame has 418 entries
print(solution.shape)

# Write your solution to a csv file with the name my_solution.csv
solution.to_csv((str(timestr + 'DTC_1.csv')), index_label = ["PassengerId"])
print(solution)


#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10 , min_samples_split = 6, random_state = 1)
my_tree_two = my_tree_two.fit(train_features, labels)

#Print the score of the new decison tree
print(my_tree_two.score(train_features, labels))

pred_3 = my_tree_two.predict(test_features)

solution3 = pd.DataFrame(pred_3, PassengerId, columns = ["Survived"])
print(solution3)

# Check that your data frame has 418 entries
print(solution3.shape)

# Write your solution to a csv file with the name my_solution.csv
solution3.to_csv((str(timestr + 'DTC_3.csv')), index_label = ["PassengerId"])
print(solution)


# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 10000, random_state = 1)
my_forest = forest.fit(train_features, labels)

# Print the score of the fitted random forest
print(my_forest.score(train_features, labels))

# Compute predictions on our test set features then print the length of the prediction vector
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

solution2 = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(solution2)

# Check that your data frame has 418 entries
print(solution2.shape)

# Write your solution to a csv file with the name my_solution.csv
solution2.to_csv((str(timestr + 'RF_2.csv')), index_label = ["PassengerId"])
print(solution2)



#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(train_features, labels))
print(my_forest.score(train_features, labels))

