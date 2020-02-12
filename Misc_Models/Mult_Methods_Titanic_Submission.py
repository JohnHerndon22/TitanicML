#RF_Titanic_Submission.py
#baseline submission at 79.904% - using RandomForestClassifer - V13 - 83% down below
# Regressor = 50%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from common import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, ShuffleSplit

def compute_family_size(df):
    # Creating new familysize column
    df['familysize']=df['SibSp']+df['Parch']+1 # the one if for the self
    df['familysize'] = df['familysize'].astype('str')
    
    return df

# def unknown_age(Age):
#     if Age==0:
#         return 1
#     else:
#         return 0

# clear the screen 
os.system('clear')

# consonants
fare_filler = -99999
age_filler = -99999

# open the training dataset - prepare dataframe
train_data = pd.read_csv(titanDir+'train.csv')
# replace NaN values with fillers
train_data = train_data.fillna({
'Age':age_filler,
'SibSp':0,
'Parch':0,
'Fare':fare_filler,
'Cabin': ''})

train_data = compute_family_size(train_data)
# train_data['len_Ticket'] = train_data.apply(lambda x: len(x['Ticket']), axis=1)
# train_data['ageUnk'] = train_data.apply(lambda x: unknown_age(x['Age']), axis=1)


# print information for review
print(train_data.columns)
print(train_data.describe())
print(train_data.head())

# prepare test data frame
test_data = pd.read_csv(titanDir+'test.csv')
test_data = test_data.fillna({
'Age':age_filler,
'SibSp':0,
'Parch':0,
'Fare':fare_filler,
'Cabin': ''})

# ready variables
y = train_data["Survived"]
features = ['Sex', 'Fare', 'Age', 'familysize', 'Pclass', 'Embarked'] # - 79.904%

# save point - ability to review training data
train_data.to_csv(titanDir+'midrun.csv')

# convert to dummies
X = pd.get_dummies(train_data[features])

# onto the test data - do the same
test_data = compute_family_size(test_data)
# test_data['len_Ticket'] = test_data.apply(lambda x: len(x['Ticket']), axis=1)
# test_data['ageUnk'] = test_data.apply(lambda x: unknown_age(x['Age']), axis=1)

# keep save point
test_data.to_csv(titanDir+'midruntest.csv')

X_test = pd.get_dummies(test_data[features])
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, oob_score=True)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, oob_score=True)

model.fit(X, y)
print(X.columns)
print(model.feature_importances_)
print("oob score: ")
print(model.oob_score_)
predictions = model.predict(X_test)
y_test = predictions

# as a check value - not used in modelling
# determine accuracy (K nearest neighbor)
#split the training data into 80/20
# X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y, test_size=0.2) 

# clf = neighbors.KNeighborsClassifier(n_jobs=-1)
# clf.fit(X, y)
# accuracy = clf.score(X_test, y_test)

# output the prediction
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(titanDir+'my_submission_v14.csv', index=False)
print("v14 submission was successfully saved - RandomForestClassifer mods")

# validate results
# shuffle_validator = ShuffleSplit(len(X), random_state=0)
shuffle_validator = ShuffleSplit(n_splits=10, random_state=0)
# print('Knn Accuracy = ', accuracy)
scores = cross_val_score(model, X, y, cv=shuffle_validator)
print("Cross Val Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
