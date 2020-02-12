#trybayes.py...

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from common import *
import pandas as pd

train_data = pd.read_csv(titanDir+'train.csv')
train_data = train_data.fillna({
'Age':0,
'SibSp':0,
'Parch':0,
'Fare':0,
'Cabin': '',
'Embarked':''})
# iris=datasets.load_iris()
# x=iris.data
# y=iris.target

# run on the training database
train_data['nEmbarked'] = train_data.apply(lambda x: convert_embarked_to_int(x['Embarked']), axis=1)
train_data['nSex'] = train_data.apply(lambda x: convert_sex_to_int(x['Sex']), axis=1)
train_data['Title'] = train_data.apply(lambda x: extract_title(x['Name'], x['Sex']), axis=1)
train_data = compute_family_size(train_data)
# features = ['nSex', 'Fare', 'Age', 'Parch', 'SibSp', 'Pclass'] 
# features = ['nSex', 'Fare', 'Age', 'Parch', 'SibSp', 'Pclass', 'nEmbarked'] 
features = ['nSex', 'Fare', 'Age', 'familysize']
X = train_data[features]
y = train_data['Survived']
print("overall survival rate: ", y.sum()/len(y))


X_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=0)
print('starting survival rate: ', y_val.sum()/len(y_val))

gnb=GaussianNB()
mnb=MultinomialNB()
y_pred_gnb=gnb.fit(X_train,y_train).predict(x_val)
cnf_matrix_gnb = confusion_matrix(y_val, y_pred_gnb)
print('predicted survival rate - Gauss: ', y_pred_gnb.sum()/len(y_pred_gnb))
print(cnf_matrix_gnb)

y_pred_mnb = mnb.fit(X_train, y_train).predict(x_val)
cnf_matrix_mnb = confusion_matrix(y_val, y_pred_mnb)
print(cnf_matrix_mnb)
print('predicted survival rate - Mulitnomial: ', y_pred_mnb.sum()/len(y_pred_mnb))

# now run the test database
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
test_data['Title'] = test_data.apply(lambda x: extract_title(x['Name'], x['Sex']), axis=1)
test_data = compute_family_size(test_data)
X_test = test_data[features]

gnb=GaussianNB()
# mnb=MultinomialNB()
y_test_gnb=gnb.fit(X_train,y_train).predict(X_test)
print('TEST -> predicted survival rate - Gauss: ', y_test_gnb.sum()/len(y_test_gnb))

# y_test_mnb = mnb.fit(X_train, y_train).predict(X_test)
# print('TEST -> predicted survival rate - Mulitnomial: ', y_test_mnb.sum()/len(y_test_mnb))

# val_mae = mean_absolute_error(y_test_gnb, y_train)
# print("Validation MAE: {:,.0f}".format(val_mae))
# shuffle_validator = ShuffleSplit(n_splits=10, random_state=0)
# scores = cross_val_score(dt_model, X, y, cv=shuffle_validator)
# print("Cross Val Score: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_test_gnb})
output.to_csv(titanDir+'my_submission_v22.csv', index=False)
print("v21 submission - Bayes Gaussian")