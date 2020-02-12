#createresultfile.py
#v2 - 2nd attempt use weighted avergae against the following factors:
# sex - 20%
# farerange - 20%
# agerange - 40%
# familysize - 40%
#
# apply individualized survivorship based on a random number

#import modules
import pandas as pan
import itertools
from common import *
import numpy as np
import os
import sys
import random

def calulate_std_survival_rate(dfresults):
    # compute the standard rate
    return round((dfresults.loc[dfresults.index,'Survived'].sum()) / (dfresults.loc[dfresults.index,'Survived'].count()),3)

# def apply_survivorship(indexArray, surviveRate, c_dftestInput):
#     # takes the indexes in the array and determines the number of survivors, applys the '1' to the survivors    
#     # determine number of survivors from total population * surivor rate 
#     numSurvivors = 0
#     if len(indexArray)!=0:
#         numSurvivors = int(round(len(indexArray) * surviveRate,0))

#     # for the first survivor number of indexs - apply the 1 to survivors
#     appValue = 1
#     counter = 1
#     for index in indexArray: 
#         c_dftestInput.loc[index,'analyzed'] = True            # apply the True to the analyze flag for each record
#         c_dftestInput.loc[index,'Survived'] = appValue
#         if counter == numSurvivors: appValue = 0
#         counter+=1
   
#     return c_dftestInput

def calc_individual_survivorship(index, genderRate, fareRate, ageRate, familyRate, c_dftestInput):
    # create an individualized survivorship
   
    #create calculated weighting for survivorship:
    dictWeights = {'Sex': .17, 'farerange': .33, 'agerange': .33,'familysize': .17}

    # calculate the expected survival rate by appling the rates to the factor
    survivalRate = (dictWeights['Sex']*genderRate)+(dictWeights['farerange']*fareRate)+(dictWeights['agerange']*ageRate)+(dictWeights['familysize']*familyRate)
    c_dftestInput.loc[index,'surviveRate']  = survivalRate

    # using random number generator - determine if the person lived or died
    c_dftestInput.loc[index,'Survived'] = int(random.random() <= survivalRate)
    
    # send the data frame back
    return c_dftestInput

#declare variables not found in common.py
# calculate average survival rate 
resDtypes = {'PassengerId':np.int32, 'Survived':np.int32, 'Pclass':str,'Name':str,'Sex':str, 'Age':str, 'SibSp':np.int32,'Parch':np.int32,'Ticket':str,'Fare':np.float,'Cabin':str,'Embarked':str,'Title':str,'Deck':str,'familysize':str,'Fare_Per_Person':np.float,'farerange':str,'agerange':str}
dfresults = pan.read_csv(titanDir+iResultsfile, dtype=resDtypes)
f = open(titanDir+'final_results_output.txt','w')
avgSurvivalRate = calulate_std_survival_rate(dfresults)
print('The average survival rate is: ' + str(avgSurvivalRate), file=f)
print('Starting.......')

#open csv for writing into dataframe - finalresults.csv - columns = passengerId, Survived
resDtypes = {'PassengerId':'int', 'Survived':'int'}
dffinalResults = pan.DataFrame(columns=['PassengerId', 'Survived'])

#open the dfsubFunct file - load parameter dataframe - primary=agerange, secondary=familysize, last=farerange
dfsubFunct = pan.read_csv(titanDir+'dfsubFunct.csv')

#pull the age range & family size values into a sub dataframe
dfsubFunctAge = dfsubFunct[dfsubFunct['description']=='agerange'].copy()

# for zero age assign the standard rate 
newRow = {'description':'agerange', 'testValuePar': np.inf,'surviveRate':avgSurvivalRate}
dfsubFunctAge = dfsubFunctAge.append(newRow, ignore_index=True)
dfsubFunctAge = dfsubFunctAge.set_index('testValuePar')

dfsubFunctFS = dfsubFunct[dfsubFunct['description']=='familysize'].copy()
dfsubFunctFS = dfsubFunctFS.set_index('testValuePar')

dfsubFunctSex = dfsubFunct[dfsubFunct['description']=='Sex'].copy()
dfsubFunctSex = dfsubFunctSex.set_index('testValuePar')

dfsubFunctFR = dfsubFunct[dfsubFunct['description']=='farerange'].copy()
# for zero Fare range assign the standard rate 
newRow = {'description':'farerange', 'testValuePar': np.inf,'surviveRate':avgSurvivalRate}
dfsubFunctFR = dfsubFunctFR.append(newRow, ignore_index=True)
dfsubFunctFR = dfsubFunctFR.set_index('testValuePar')

#open the test file - for analysis 
# dftestInput = pan.read_csv(titanDir+'train.csv')
dftestInput = pan.read_csv(titanDir+'test.csv')


#prep the test file - create columns for age range, family size and farerange - create blanks where none exists
# Creating new familysize column
dftestInput['familysize']=dftestInput['SibSp']+dftestInput['Parch']+1 # the one if for the self
dftestInput['familysize'] = dftestInput['familysize'].astype('str')

# make a histogram of the ages - 0-10, 10-20, 21-30 and so on 
dftestInput = dftestInput.set_index('Age', drop=False)
dftestInput['agerange'] = pan.cut(dftestInput['Age'], agebins, labels=agelabels) 

dftestInput = dftestInput.set_index('Fare', drop=False)
dftestInput['farerange'] = pan.cut(dftestInput['Fare'], farebins, labels=farelabels)
dftestInput['agerange'] = dftestInput['agerange'].cat.add_categories(np.inf)
dftestInput['farerange'] = dftestInput['farerange'].cat.add_categories(np.inf)
dftestInput = dftestInput.fillna({'agerange':np.inf, 'farerange':np.inf})

dftestInput = dftestInput.set_index('PassengerId', drop=False)

# create blank survive rate column - this will hold the prediction factor
dftestInput['surviveRate']=.000

print('Total passengers to evaluate: ' + str(len(dftestInput.index)), file=f)

# to be deleted
# for each age range - take the survivability rate assign the correct number of people as survivors, assign the correct number of people as deceased
# dftestInput = apply_survivorship(dftestInput[dftestInput['agerange']==ageranger.testValuePar].index, ageranger.surviveRate, dftestInput.copy())
    
#now write the status and passenger number to the results file
dictresults = {'PassengerId':0,'Survived':0}

for index, passenger in dftestInput.iterrows():

    print('evaluating passenger-->' + str(passenger['PassengerId']))
     # determine the sex value
    genderRate = dfsubFunctSex.loc[passenger.Sex,'surviveRate']
   
    # determine the age range - avg rate for 0 
    ageRate = dfsubFunctAge.loc[passenger.agerange,'surviveRate']

    # determine the fare value
    fareRate = dfsubFunctFR.loc[passenger.farerange,'surviveRate']
    
    # determine the family size value 
    try: familyRate = dfsubFunctFS.loc[passenger.familysize,'surviveRate']
    except: familyRate = avgSurvivalRate
    
    # copy the dataframe
    c_dftestInput = dftestInput.copy()

    # call the calc survivor ship function
    dftestInput = calc_individual_survivorship(index, genderRate, fareRate, ageRate, familyRate, c_dftestInput)

    dictresults['PassengerId'] = passenger['PassengerId']
    dictresults['Survived'] = dftestInput.loc[index,'Survived']
    dffinalResults = dffinalResults.append(dictresults, ignore_index=True)
    dictresults = {'PassengerId':0,'Survived':0}

dffinalResults['PassengerId'] = dffinalResults['PassengerId'].astype('int')
dffinalResults['Survived'] = dffinalResults['Survived'].astype('int')

# print stats about the test file:
print('total males: ' + str(dftestInput[dftestInput['Sex']=='male'].Survived.count())+ '   total females: '+str(dftestInput[dftestInput['Sex']=='female'].Survived.count()))
print('age distribution: ')
table = pan.pivot_table(dftestInput, values='Survived', index=['agerange'], margins=True, aggfunc='count')
print(table)
print('fare distribution: ')
table = pan.pivot_table(dftestInput, values='Survived', index=['farerange'], margins=True, aggfunc='count')
print(table)
print('family size distribution: ')
table = pan.pivot_table(dftestInput, values='Survived', index=['familysize'], margins=True, aggfunc='count')
print(table)

#print stats about the results file
print('total passengers: '+ str(dffinalResults.Survived.count()), file=f)
print('total survived: '+ str(dffinalResults.Survived.sum()), file=f)
print('total dead: ' + str(dffinalResults.Survived.count() - dffinalResults.Survived.sum()), file=f)
print('survivor rate: ' + str(dffinalResults.Survived.sum() / dffinalResults.Survived.count()), file=f)
print('Done.......')

# create the final results file
print(dffinalResults['PassengerId'].count(), file=f)
dffinalResults.to_csv(titanDir+'final_results-'+processdate+'.csv', index=False)
