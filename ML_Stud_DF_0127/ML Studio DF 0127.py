#RF_Titanic_Submission.py
#baseline submission at 79.904% - using RandomForestClassifer - V13

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from common import *
import os
import random

def make_prediction(Score):
    Score = Score * 100
    if Score >= 100:
        return 1
    else:
        arr_rander = np.arange(7000)
        for c in range(7000):
            arr_rander[c] = random.randrange(100)
        rander = np.average(arr_rander)
        
        # print(rander) 
        
        if rander < Score:
            return 1            # passenger lives
        else:
            return 0            # passenger dies


# clear the screen 
os.system('clear')
titanDir = '/Users/johncyclist22/Documents/ML_Competitions/Titanic/Data/Submissions/'

# open the training dataset - prepare dataframe
predict_data = pd.read_csv('My Submission DF 0127.csv')
predict_outcome = pd.DataFrame()


# print information for review
print(predict_data.columns)
print(predict_data.describe())
print(predict_data.head())

# for counter in range(10):

predict_outcome['PassengerId'] = predict_data['PassengerId']
predict_outcome['Survived'] = predict_data.apply(lambda x: make_prediction(x['Scored Probabilities']), axis=1)
# print('Run #: '+str(counter+1)+ "--> Survivors: " + str(predict_outcome['Survived'].sum()) + " out of " + str(predict_outcome['Survived'].count()) + " Rate--> " + str(predict_outcome['Survived'].sum() / predict_outcome['Survived'].count() )   ) 
print("--> Survivors: " + str(predict_outcome['Survived'].sum()) + " out of " + str(predict_outcome['Survived'].count()) + " Rate--> " + str(predict_outcome['Survived'].sum() / predict_outcome['Survived'].count() )   ) 

print(predict_outcome.head())
print(predict_outcome.describe())


predict_outcome.to_csv(titanDir+'my_submission_v34.csv', index=False)
print("v34 submission was successfully saved - Decision Forest Azure ML")
