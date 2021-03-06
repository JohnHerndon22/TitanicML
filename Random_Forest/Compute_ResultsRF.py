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
        arr_rander = np.arange(10000)
        for c in range(10000):
            arr_rander[c] = random.randrange(100)
        rander = np.average(arr_rander)
        
        # print(rander) 
        
        if rander < Score:
            return 1            # passenger lives
        else:
            return 0            # passenger dies


# clear the screen 
os.system('clear')

# open the training dataset - prepare dataframe
predict_data = pd.read_csv(outputDir+'t_randomforest.csv')
predict_outcome = pd.DataFrame()


# print information for review
print(predict_data.columns)
print(predict_data.describe())
print(predict_data.head())

for counter in range(5):

    predict_outcome['PassengerId'] = predict_data['PassengerId']
    predict_outcome['Survived'] = predict_data.apply(lambda x: make_prediction(x['Survived']), axis=1)
    print('Run #: '+str(counter+1)+ "--> Survivors: " + str(predict_outcome['Survived'].sum()) + " out of " + str(predict_outcome['Survived'].count()) + " Rate--> " + str(predict_outcome['Survived'].sum() / predict_outcome['Survived'].count() )   ) 
    

print(predict_outcome.head())
print(predict_outcome.describe())


predict_outcome.to_csv(outputDir+'my_submission_v51.csv', index=False)
print("v51 submission was successfully saved - Random Forest Probs")
