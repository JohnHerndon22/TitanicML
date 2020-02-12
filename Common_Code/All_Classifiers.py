import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from common import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, ShuffleSplit
import random

# def evaluate_score(model, X, real_prediction):
#     # prints assessment (scoring) of the prediction
#     shuffle_validator = ShuffleSplit(n_splits=10, random_state=0)
#     scores = cross_val_score(model, X, real_prediction, cv=shuffle_validator)
#     print("Cross Val Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
    
#     return True
def gen_rand():
    randoms = np.zeros(shape=(100))
    for i in range(100):
        randoms[i] = random.random()
    return randoms.mean()


def make_predictions(classifier, train_X, train_y, test_X):
    # from training data and test features, returns predictive % and prediction into array
    clf = classifier
    clf.fit(train_X, train_y)
    real_predictions = clf.predict(test_X)
    per_predictions = clf.predict_proba(test_X)
    per_predictions1 = []
    for x in per_predictions: per_predictions1 = np.append(per_predictions1, x[1]) 
    
    shuffle_validator = ShuffleSplit(n_splits=10, random_state=0)
    # scores = cross_val_score(clf, test_X, real_predictions, cv=shuffle_validator)
    # print("Cross Val Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
    
    # evaluate_score(clf, test_X, real_predictions)
    
    return per_predictions1, real_predictions

def write_results_to_files(key, per_predictions, real_predictions, PassengerIds, real_results, per_results):
    # writes results into two files
    fileSuffix = "_preds"
    dirsuffix = "methodpredictions/"
    output = pd.DataFrame({'PassengerId': PassengerIds, 'Survived': real_predictions})
    per_output = pd.DataFrame({'PassengerId': PassengerIds, 'Survived': per_predictions})
    output.to_csv(titanDir+dirsuffix+key+fileSuffix+'_real.csv', index=False)
    per_output.to_csv(titanDir+dirsuffix+key+fileSuffix+'_per.csv', index=False)
    real_results[key] = real_predictions
    per_results[key] = per_predictions

    return True 

# main
# initialize variables
# nfeatures = ['nSex', 'Fare', 'Age', 'familysize', 'Pclass', 'nEmbarked']  765
nfeatures = ['nSex', 'Fare', 'familysize', 'Pclass', 'nEmbarked'] # 77%
# nfeatures = ['nSex', 'Fare', 'familysize', 'Pclass'] # 76.5%
nfeatures = ['nSex', 'Fare', 'Pclass'] # 76.5%
# features = ['Sex', 'Fare', 'Age', 'familysize', 'Pclass', 'Embarked']
overallResults = {'predictedSurvival': .000}
per_predictions = []
real_predictions = []
PassengerIds = []
keys=[]
dirsuffix = "methodpredictions/"

# open the training and test datasets
train_data = pd.read_csv(titanDir+'train.csv')
test_data = pd.read_csv(titanDir+'test.csv')
per_results = pd.DataFrame()
real_results = pd.DataFrame()

# clean the data
train_data = train_data.fillna({'Age':age_filler,'SibSp':0,'Parch':0,'Fare':fare_filler,'Cabin': ''})
test_data = test_data.fillna({'Age':age_filler,'SibSp':0,'Parch':0,'Fare':fare_filler,'Cabin': ''})
# PassengerIds = test_data['PassengerIds']

# compute (i.e. family size) 
train_data = compute_family_size(train_data)
train_data['nEmbarked'] = train_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
train_data['nSex'] = train_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)

# same for the test data
test_data = compute_family_size(test_data)
test_data['nEmbarked'] = test_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
test_data['nSex'] = test_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)

# prepare test and training arrays
X = train_data[nfeatures]
y = train_data['Survived']
test_X = test_data[nfeatures]
print("overall starting survival rate(test data): ", y.sum()/len(y))

per_results['PassengerId'] = test_data.PassengerId
real_results['PassengerId'] = test_data.PassengerId

for key in classifiers:
    print("running....", key)
    # call make predictions
    per_predictions, real_predictions = make_predictions(classifiers[key], X, y, test_X)
    print('prediction is: ', str(real_predictions.sum() / len(real_predictions) ))

    # keys = np.append(keys, key)

    # write the results to the files
    write_results_to_files(key, per_predictions, real_predictions, test_data.PassengerId, real_results, per_results)
    per_predictions = []
    real_predictions = []

# write overallResults to file
real_results.to_csv(titanDir+dirsuffix+'overall_real.csv', index=False)
per_results.to_csv(titanDir+dirsuffix+'overall_per.csv', index=False)

passResult = {'PassengerId': 0, 'Survived': 0}
all_results = pd.DataFrame(columns = ['PassengerId','Survived'], dtype=np.int32)
per_results = per_results.set_index(['PassengerId'])
per_results['avgSurvival'] = per_results.mean(axis=1)    
  
for id, passenger in per_results.iterrows():
    action = gen_rand()
    passResult['PassengerId'] = int(id)
    if action < passenger.avgSurvival: passResult['Survived'] = 1
    all_results = all_results.append(passResult, ignore_index=True)
    passResult = {'PassengerId': 0, 'Survived': 0}

all_results.to_csv(titanDir+dirsuffix+'overall_avg.csv', index=False)
print('avg survival: ', str(int(all_results['Survived'].sum()) / len(all_results['Survived'])))


        

    
    
