#Titanic_AdaBoost.py
# Make Titanic Predictions using the AdaBoost prediction algortyhim from SKLearn

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from common import *
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import preprocessing

def normalize_fare(fareArray):
    xArray = [[]]
    xArray[0] = fareArray
    transformer = preprocessing.Normalizer().fit(xArray)
    txArray = transformer.transform(xArray)
    # must pull single cell out of the array
    return txArray[0]


def convert_embarked_to_int(embarked):
    if embarked == "S": 
        Nembarked = 1
    elif embarked == "C":
        Nembarked = 2
    else:
        Nembarked = 3
    return Nembarked

def convert_sex_to_int(Sex):
    if Sex == "male": 
        NSex = 1
    else:
        NSex = 0
    return NSex

def compute_family_size(df):
    # Creating new familysize column
    df['familysize']=df['SibSp']+df['Parch']+1 # the one if for the self
    df['familysize'] = df['familysize'].astype('str')
    
    return df

def open_data_file(filetoOpen):
    df = pd.read_csv(titanDir+filetoOpen)
    df = df.fillna({
    'Age':-99999,
    'SibSp':0,
    'Parch':0,
    'Fare':-99999,
    'Cabin': '',
    'Embarked':''})

    df = compute_family_size(df)
    df['nEmbarked'] = df.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
    df['nSex'] = df.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
    df['norm_Fare'] = normalize_fare(df['Fare'])
    dataArray = df[features].to_numpy()
    return dataArray

def get_passenger_ids(testfile):
    df = pd.read_csv(titanDir+testfile)
    testArray = df['PassengerId'].to_numpy()
    return testArray


def get_survival_stats():
    df = pd.read_csv(titanDir+'train.csv')
    return df["Survived"]

# main
features = ['nSex', 'Fare', 'Age', 'familysize', 'Pclass', 'nEmbarked']

trainArray = open_data_file('train.csv')
testArray = open_data_file('test.csv')
test_data = pd.read_csv(titanDir+'test.csv')
ResultsArray = get_passenger_ids('test.csv')

X = pd.DataFrame(trainArray, columns=features)
y = get_survival_stats()

encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(y))

# train_X, val_X, train_y, val_y = train_test_split(X, binary_encoded_y, random_state=1, test_size = .2)

# print('orginal val surivival rate: '+str(val_y.values.sum()/len(val_y)))

classifier = AdaBoostClassifier(
    RandomForestClassifier(max_depth=1),
    n_estimators=100, learning_rate=1, random_state=42)
classifier.fit(X, binary_encoded_y)
predictions = classifier.predict(testArray)
print(classifier.base_estimator_)
print(classifier.estimators_)
print(classifier.classes_)
print(classifier.n_classes_)
print(classifier.estimator_weights_)
print(classifier.estimator_errors_)
print(classifier.feature_importances_)



print(predictions)
print('training DS surivival rate: '+str(y.values.sum()/len(y)))
print('predicted surivival rate: '+str(predictions.sum()/len(predictions)))
# print(confusion_matrix(y, predictions))
per_predictions = classifier.predict_proba(testArray)
per_predictions1 = []
for x in per_predictions: per_predictions1 = np.append(per_predictions1, x[1]) 
per_output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': per_predictions1})

# shuffle_validator = ShuffleSplit(n_splits=10, random_state=42)
# scores = cross_val_score(classifier, testArray, predictions, cv=shuffle_validator)
# print("Cross Validation Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

# output the prediction
output = pd.DataFrame({'PassengerId': ResultsArray, 'Survived': predictions})
output.to_csv(titanDir+'my_submission_v19.csv', index=False)
per_output.to_csv(titanDir+'t_adaboost.csv', index=False)
print("v19 submission was successfully submitted - AdaBoostClassifier, RandomForestClassifier")