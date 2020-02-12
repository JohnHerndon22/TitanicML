#createresultfile.py

#import modules
import pandas as pan
import itertools
from common import *
import numpy as np
import os
import sys

def calulate_std_survival_rate(dfresults):
    # compute the standard rate
    return round((dfresults.loc[dfresults.index,'Survived'].sum()) / (dfresults.loc[dfresults.index,'Survived'].count()),3)

def apply_survivorship(indexArray, surviveRate, c_dftestInput):
    # takes the indexes in the array and determines the number of survivors, applys the '1' to the survivors    
    # determine number of survivors from total population * surivor rate
    numSurvivors = 0
    if len(indexArray)!=0:
        numSurvivors = int(round(len(indexArray) * surviveRate,0))

    # for the first survivor number of indexs - apply the 1 to survivors
    appValue = 1
    counter = 1
    for index in indexArray: 
        c_dftestInput.loc[index,'analyzed'] = True            # apply the True to the analyze flag for each record
        c_dftestInput.loc[index,'Survived'] = appValue
        if counter == numSurvivors: appValue = 0
        counter+=1
   
    return c_dftestInput

#declare variables not found in common.py
# calculate average survival rate 
resDtypes = {'PassengerId':np.int32, 'Survived':np.int32, 'Pclass':str,'Name':str,'Sex':str, 'Age':str, 'SibSp':np.int32,'Parch':np.int32,'Ticket':str,'Fare':np.float,'Cabin':str,'Embarked':str,'Title':str,'Deck':str,'familysize':str,'Fare_Per_Person':np.float,'farerange':str,'agerange':str}
dfresults = pan.read_csv(titanDir+iResultsfile, dtype=resDtypes)
f = open(titanDir+'final_results_output.txt','w')
avgSurvivalRate = calulate_std_survival_rate(dfresults)
print('The average survival rate is: ' + str(avgSurvivalRate), file=f)

#open csv for writing into dataframe - finalresults.csv - columns = passengerId, Survived
resDtypes = {'PassengerId':'int', 'Survived':'int'}
dffinalResults = pan.DataFrame(columns=['PassengerId', 'Survived'])

#open the dfsubFunct file - load parameter dataframe - primary=agerange, secondary=familysize, last=farerange
dfsubFunct = pan.read_csv(titanDir+'dfsubFunct.csv')

#pull the age range & family size values into a sub dataframe
dfsubFunctAge = dfsubFunct[dfsubFunct['description']=='agerange'].copy()
dfsubFunctFS = dfsubFunct[dfsubFunct['description']=='familysize'].copy()

#open the test file - for analysis 
dftestInput = pan.read_csv(titanDir+'test.csv')
dftestInput['analyzed'] = False

#prep the test file - create columns for age range, family size and farerange - create blanks where none exists
# Creating new familysize column
dftestInput['familysize']=dftestInput['SibSp']+dftestInput['Parch']+1 # the one if for the self

# make a histogram of the ages - 0-10, 10-20, 21-30 and so on 
dftestInput = dftestInput.set_index('Age', drop=False)
dftestInput['agerange'] = pan.cut(dftestInput['Age'], agebins, labels=agelabels)
dftestInput = dftestInput.set_index('PassengerId', drop=False)

print('Total passengers to evaluate: ' + str(len(dftestInput.index)), file=f)

#for each consistuency - report the number of people impacted
#loop thru all of the agerange values
for index, ageranger in dfsubFunctAge.iterrows():

    print('Analyzing age range: ' + ageranger.testValuePar + '    passengers--> ' + str(len(dftestInput[dftestInput['agerange']==ageranger.testValuePar].index)), file=f)
    
    #for each age range - take the survivability rate assign the correct number of people as survivors, assign the correct number of people as deceased
    dftestInput = apply_survivorship(dftestInput[dftestInput['agerange']==ageranger.testValuePar].index, ageranger.surviveRate, dftestInput.copy())
    
print('Total passengers left to evaluate: ' + str(len(dftestInput[dftestInput['analyzed']==False].index)), file=f)

#are there any people with blank age range but family sizes > 0?  if so - loop thru each family size touching only the people not included above
for index, FSranger in dfsubFunctFS.iterrows():
    if FSranger.testValuePar !='inf':
        print('Analyzing family sizes: ' + FSranger.testValuePar + '    passengers--> ' + str(len(dftestInput[(dftestInput['familysize']==int(FSranger.testValuePar))&(dftestInput['analyzed']==False)].index)), file=f)        
        dftestInput = apply_survivorship(dftestInput[(dftestInput['familysize']==int(FSranger.testValuePar))&(dftestInput['analyzed']==False)].index, FSranger.surviveRate, dftestInput.copy())
        
print('Total passengers left to evaluate: ' + str(len(dftestInput[dftestInput['analyzed']==False].index)), file=f)

# apply the standard rate to remaining passengers
dftestInput = apply_survivorship(dftestInput[dftestInput['analyzed']==False].index, avgSurvivalRate, dftestInput.copy())

print('Total passengers left to evaluate: ' + str(len(dftestInput[dftestInput['analyzed']==False].index)), file=f)

#now write the status and passenger number to the results file
dictresults = {'PassengerId':0,'Survived':0}

for index, record in dftestInput.iterrows():
    dictresults['PassengerId'] = record['PassengerId']
    dictresults['Survived'] = record['Survived']
    dffinalResults = dffinalResults.append(dictresults, ignore_index=True)
    dictresults = {'PassengerId':0,'Survived':0}

dffinalResults['PassengerId'] = dffinalResults['PassengerId'].astype('int')
dffinalResults['Survived'] = dffinalResults['Survived'].astype('int')

# dffinalResults['Survived'] = dftestInput['Survived']
print(dffinalResults, file=f)
dffinalResults.to_csv(titanDir+'final_results.csv', index=False)

#print stats about the results file
print('total passengers: '+ str(dffinalResults.Survived.count()), file=f)
print('total survived: '+ str(dffinalResults.Survived.sum()), file=f)
print('total dead: ' + str(dffinalResults.Survived.count() - dffinalResults.Survived.sum()), file=f)
print('survivor rate: ' + str(dffinalResults.Survived.sum() / dffinalResults.Survived.count()), file=f)
print('Done.......')
