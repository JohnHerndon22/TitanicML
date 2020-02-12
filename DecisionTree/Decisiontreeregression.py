#forestmodel.py

# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
import os
from common import *

def standardize_data(xArray):
    # xArray = [[]]
    # xArray[0] = fareArray
    # xArray[0] = fareArray
    print(xArray)
    transformer = preprocessing.RobustScaler().fit(xArray)
    # transformer = preprocessing.Normalizer().fit(xArray)
    txArray = transformer.transform(xArray)


    print(txArray)
    # must pull single cell out of the array
    return txArray

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


# clear the screen 
os.system('clear')

train_data = pd.read_csv(titanDir+'train.csv')
train_data = train_data.fillna({
'Age':0,
'SibSp':0,
'Parch':0,
'Fare':0,
'Cabin': '',
'Embarked':''})

# Create target object and call it y
y = train_data['Survived']
print("overall survival rate: ", y.sum()/len(y))
train_data['nEmbarked'] = train_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
train_data['nSex'] = train_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
train_data = compute_family_size(train_data)


# Create X
# features = ['nSex', 'Fare', 'Age', 'Parch', 'SibSp', 'Pclass', 'familysize', 'nEmbarked'] 
features = ['nSex', 'Fare', 'Age', 'Parch', 'SibSp', 'Pclass', 'nEmbarked'] 

# features = ['nSex', 'Fare', 'Age', 'Parch', 'SibSp', 'Pclass', 'nEmbarked'] 
X = train_data[features]
# X = standardize_data(X)   # sends the data down

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size = .2)
print("training data")
print(train_X.describe())
print('validation data')
print(val_X.describe())
print('starting survival rate: ', val_y.sum()/len(val_y))
# Specify Model
# Fit Model
dt_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
dt_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = dt_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE: {:,.0f}".format(val_mae))
shuffle_validator = ShuffleSplit(n_splits=10, random_state=0)
scores = cross_val_score(dt_model, X, y, cv=shuffle_validator)
print("Cross Val Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
print('predicted survival rate: ', val_predictions.sum()/len(val_predictions))
# quit()

test_data = pd.read_csv(titanDir+'test.csv')
test_data = test_data.fillna({
'Age':0,
'SibSp':0,
'Parch':0,
'Fare':0,
'Cabin': '',
'Embarked':''})

test_data['nEmbarked'] = test_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
test_data['nSex'] = test_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
test_data = compute_family_size(test_data)
test_X = test_data[features]

# t_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
t_model = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)

t_model.fit(X, y)

# Make validation predictions and calculate mean absolute error
predictions = t_model.predict(test_X)
per_predictions = t_model.predict_proba(test_X)
per_predictions1 = []
for x in per_predictions: per_predictions1 = np.append(per_predictions1, x[1]) 

# val_mae = mean_absolute_error(predictions, y)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(titanDir+'my_submission_vXX.csv', index=False)
per_output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': per_predictions1})
per_output.to_csv(titanDir+'t_decisiontree.csv', index=False)


print("vXX submission was successfully saved - Decision tree")

