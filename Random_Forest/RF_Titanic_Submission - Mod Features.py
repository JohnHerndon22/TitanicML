#RF_Titanic_Submission.py
#baseline submission at 79.904% - using RandomForestClassifer - V13
#V41 submission with features: Nsex, Pclass, Deck, mFare, Nembarked - based on feature testing in Azure ML Studio.
#file has been prepped by prep_input_file.py


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from common import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
    
# clear the screen 
os.system('clear')

# open the training dataset - prepare dataframe
train_data = pd.read_csv(titanDir+'prep_train_head.csv')

# print information for review
print(train_data.columns)
print(train_data.describe())
print(train_data.head())

# prepare test data frame
test_data = pd.read_csv(titanDir+'prep_test_head.csv')

# ready variables - get survivor status for the training set
y = train_data["Survived"]
y_true = y

features = ['Nsex', 'Pclass', 'Deck', 'mFare', 'Nembarked']     # these features with n's brought 79.425% 


# features = ['Sex', 'Pclass', 'Deck', 'Fare', 'Embarked']    # goes down to 75598
# features = ['female_uc', 'Nsex', 'Pclass', 'Deck', 'Fare', 'Wealthy', 'Nembarked']     # loaded up with co-linear variables - 76.076%
# features = ['Nsex', 'Pclass', 'Deck', 'mFare', 'Nembarked', 'familysize', 'Age']    # down to 67.942% !
# features = ['Sex', 'Fare', 'Age', 'familysize', 'Pclass', 'Embarked'] # - 79.904% - this was produced earlier

# save point - ability to review training data
train_data.to_csv(titanDir+'midrun.csv')

# convert to dummies - both training and test sets
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# build the trained model model 
model = RandomForestClassifier(n_estimators=32, max_depth=32, min_samples_split=2, random_state=1, oob_score=True)
model.fit(X, y)

y_pred = model.predict(X)

# make predictions based on the test set
predictions = model.predict(X_test)
per_predictions = model.predict_proba(X_test)
per_predictions1 = []
for x in per_predictions: per_predictions1 = np.append(per_predictions1, x[1]) 

# output the prediction
print("predictions is: " + str(predictions.sum() / len(predictions)))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
per_output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': per_predictions1})
output.to_csv(outputDir+'my_submission_v41.csv', index=False)
per_output.to_csv(outputDir+'t_randomforest.csv', index=False)
print("v41 submission was successfully saved - RandomForestClassifer - Modified Features Azure ML - Baseline")

# save off predictions for analysis
output_review = test_data
output_review['pred_Survived'] = predictions
print(output_review.head())
output_review.to_csv(titanDir+'test_pred_anal.csv', index=False, header=True)

# validate results
# shuffle_validator = ShuffleSplit(len(X), random_state=0)
shuffle_validator = ShuffleSplit(n_splits=10, random_state=0)
# print('Knn Accuracy = ', accuracy)
scores = cross_val_score(model, X, y, cv=shuffle_validator)
print("Cross Val Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("F1 Scoring - Metrics Training Set")
F1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print('Precision: ', F1[0])
print('Recall: ', F1[1])
print('F1: ', F1[2])
