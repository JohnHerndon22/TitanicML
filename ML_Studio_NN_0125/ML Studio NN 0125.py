#RF_Titanic_Submission.py
#baseline submission at 79.904% - using RandomForestClassifer - V13

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from common import *
import os
import random

def make_prediction(Score):
    Score = Score * 100
    if Score >= 100:
        return 1
    else:
        rander = random.randrange(100)
        print(rander) 
        if rander < Score:
            return 1            # passenger lives
        else:
            return 0            # passenger dies


# clear the screen 
os.system('clear')


# open the training dataset - prepare dataframe
predict_data = pd.read_csv(titanDir+'My Submission Input 0125 NN.csv')
predict_outcome = pd.DataFrame()


# print information for review
print(predict_data.columns)
print(predict_data.describe())
print(predict_data.head())

predict_outcome['PassengerId'] = predict_data['PassengerId']
predict_outcome['Survived'] = predict_data.apply(lambda x: make_prediction(x['Scored Probabilities']), axis=1)

print(predict_outcome.head())
print(predict_outcome.describe())


# print("prediction is: " + str(predict_outcome.Survived.sum() / predict_outcome.Survived.count())
# output = pd.DataFrame({'PassengerId': predict_outcome.PassengerId, 'Survived': predict_outcome.Survived})
predict_outcome.to_csv(titanDir+'my_submission_v30.csv', index=False)
print("v30 submission was successfully saved - Neutral Network Azure ML")
