# random_forest_confusion.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from common import *

train_data = pd.read_csv(titanDir+'prep_train_head.csv')
# features = ['Nsex', 'Pclass', 'Deck', 'mFare', 'Nembarked'] 
features = ['female_uc' , 'Nsex' , 'Pclass' , 'Deck' , 'mFare' , 'Fare' , 'Wealthy' , 'Nembarked' , 'Parch' , 'Age' , 'mAge' , 'SibSp' , 'familysize' , 'Title' , 'ticketType']

df = pd.DataFrame(train_data, columns=features)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .70
df['passengers'] = pd.factorize(iris.target, iris.target_names)
print(df.head())

train, cv = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)

preds = iris.target_names[clf.predict(test[features])]
pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])

