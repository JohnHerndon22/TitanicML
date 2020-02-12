#titanexample.py
#v5 - add more features - plclass and embarked
#v6 - grab the family name and grab the title
# - v12 - ['nSex', 'Fare', 'Age', 'familysize, 'Pclass', 'nEmbarked'] - normalized fare - 76.076% - normalize not necessary and was harmful to the model
# - v11 - ['Sex', 'Fare', 'Age', 'SibSp', 'Parch', 'Pclass', 'Embarked'] - 78.47% 
# - v10 features = ['Sex', 'Fare', 'Age', 'familysize', 'Pclass', 'Embarked'] - 79.904%
# - v8 - agerange - reduced score to 78%
# - v7 - reduced score - 'Deck']
# - v6 - had no impact:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from common import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split


def normalize_fare(fareArray):
    xArray = [[]]
    xArray[0] = fareArray
    transformer = preprocessing.Normalizer().fit(xArray)
    txArray = transformer.transform(xArray)
    # must pull single cell out of the array
    return txArray[0]



def compute_family_size(df):
    # Creating new familysize column
    df['familysize']=df['SibSp']+df['Parch']+1 # the one if for the self
    df['familysize'] = df['familysize'].astype('str')
    
    return df

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


def pull_deck(Cabin):
    if len(Cabin) > 0:
        Deck=Cabin[:1]
        if Deck=="T": Deck="G"
    else:
        return ''
    return Deck

def extract_family_name(Name):
    return Name[0:Name.find(', ')]

def extract_title(Name, sex):
    title =  Name[Name.find(', ')+2:-(len(Name) - Name.find('. ', Name.find(', ')+2))]
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['Countess', 'Mme', 'the Countess','Lady', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if sex=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

# clear the screen 
os.system('clear')

# prepare training data frame
fare_filler = -99999
avg_fare = 32.7
age_filler = -99999
avg_age = 30

train_data = pd.read_csv(titanDir+'train.csv')
train_data = train_data.fillna({
'Age':age_filler,
'SibSp':0,
'Parch':0,
'Fare':fare_filler,
'Cabin': ''})

# add title column
train_data['Title'] = train_data.apply(lambda x: extract_title(x['Name'], x['Sex']), axis=1)
train_data['Family Name'] = train_data.apply(lambda x: extract_family_name(x['Name']), axis=1)
train_data['Deck'] = train_data.apply(lambda x: pull_deck(x['Cabin']), axis=1)

train_data = train_data.set_index('Age', drop=False)
train_data['agerange'] = pan.cut(train_data['Age'], agebins, labels=agelabels)

train_data['nEmbarked'] = train_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
train_data['nSex'] = train_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
train_data['norm_Fare'] = normalize_fare(train_data['Fare'])
train_data = compute_family_size(train_data)
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

# add title column
test_data['Title'] = test_data.apply(lambda x: extract_title(x['Name'], x['Sex']), axis=1)
test_data['Family Name'] = test_data.apply(lambda x: extract_family_name(x['Name']), axis=1)
test_data['Deck'] = test_data.apply(lambda x: pull_deck(x['Cabin']), axis=1)

# make a histogram of the ages - 0-10, 10-20, 21-30 and so on 
test_data = test_data.set_index('Age', drop=False)
test_data['agerange'] = pan.cut(test_data['Age'], agebins, labels=agelabels)

test_data['nEmbarked'] = test_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
test_data['nSex'] = test_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
test_data['norm_Fare'] = normalize_fare(test_data['Fare'])
test_data = compute_family_size(test_data)

# ready variables
y = train_data["Survived"]
features = ['nSex', 'norm_Fare', 'Age', 'familysize', 'Pclass', 'nEmbarked']

# save ppoint
train_data.to_csv(titanDir+'midrun.csv')

# convert to dummies
X = pd.get_dummies(train_data[features])

# develop features for farerange, agerane, and familysize - test.csv
c_test_data = test_data.copy()
test_data = compute_family_size(c_test_data)
test_data.to_csv(titanDir+'midruntest.csv')

X_test = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# determine accuracy (K nearest neighbor)
#split the training data into 80/20
X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y, test_size=0.2) 
clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train1, y_train1)
accuracy = clf.score(X_train2, y_train2)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(titanDir+'my_submission_v12.csv', index=False)
print("v12 submission was successfully saved - normalized data")
print('Accuracy = ', accuracy)


# Any results you write to the current directory are saved as output.