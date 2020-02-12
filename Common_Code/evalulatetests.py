#evaluatetests.py
#evaluate permutations of each type of test against titanic data

import pandas as pan
import itertools
from common import *
from scipy import stats
import numpy as np
import os
import statistics as st

def calulate_std_survival_rate(dfresults):
    # compute the standard rate
    return round((dfresults.loc[dfresults.index,'Survived'].sum()) / (dfresults.loc[dfresults.index,'Survived'].count()),3)

# def calc_overall_stats_tests(realHy, nullHy):
#     # calculate pvalue and correlation 
#     return stats.pearsonr(realHy, nullHy)

def create_match_tests(parentTest, childTest, testNum, testStr, dfresults, avgSurvivalRate,f):
    # pull the two arrays in - parent-child and send matching dataframe back with(example):
    c_dfsubFunct = pan.DataFrame(columns=['testNum', 'description', 'survived', 'dead', 'total','surviveRate', 'nullsurvived', 'condition','MTtestPar', 'testValuePar','MTtestChild', 'testValueChild'])
    dictsubFunct = {'testNum':testNum, 'description':testStr, 'survived':0, 'dead':0, 'total':0,'surviveRate':0, 'nullsurvived':0, 'condition':'','MTtestPar':'', 'testValuePar':'','MTtestChild':'', 'testValueChild':''}

    # if child test is empty add one blank record
    if len(childTest['testValue']) == 0:
        childTest['testValue'] = ('blank',)
        childTest['majorTest'] = 'blank'

    for PTvalue in parentTest['testValue']:
        dictsubFunct['MTtestPar'] = parentTest['majorTest']
        dictsubFunct['testValuePar'] = PTvalue   
        for CDvalue in childTest['testValue']:
            dictsubFunct['MTtestChild'] = childTest['majorTest']
            dictsubFunct['testValueChild'] = CDvalue   

            # compute the condition and place into the dataframe
            if childTest['majorTest'] != 'blank':
                dictsubFunct['survived'] = dfresults[(dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar']) & (dfresults[dictsubFunct['MTtestChild']]==dictsubFunct['testValueChild'])].Survived.sum()
                dictsubFunct['nullsurvived'] = round((dfresults[(dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar']) & (dfresults[dictsubFunct['MTtestChild']]==dictsubFunct['testValueChild'])].Survived.count())*avgSurvivalRate,0)
                dictsubFunct['surviveRate'] = round(dfresults[ (dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar']) & (dfresults[dictsubFunct['MTtestChild']]==dictsubFunct['testValueChild']) ].Survived.mean(),4)
                dictsubFunct['dead'] = dfresults[ (dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar']) & (dfresults[dictsubFunct['MTtestChild']]==dictsubFunct['testValueChild']) ].Survived.count() - dfresults[ (dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar']) & (dfresults[dictsubFunct['MTtestChild']]==dictsubFunct['testValueChild']) ].Survived.sum()
                dictsubFunct['total'] = dfresults[ (dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar']) & (dfresults[dictsubFunct['MTtestChild']]==dictsubFunct['testValueChild']) ].Survived.count()
                dictsubFunct['condition'] = "(dfresults['"+ dictsubFunct['MTtestPar'] + "']=='" + dictsubFunct['testValuePar'] + "') & (dfresults['"+ dictsubFunct['MTtestChild'] + "']=='" + dictsubFunct['testValueChild'] + "')" 
            else:
                dictsubFunct['condition'] = "dfresults['"+ dictsubFunct['MTtestPar'] + "']=='" + dictsubFunct['testValuePar'] +"'"
                dictsubFunct['survived'] = dfresults[ dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar'] ].Survived.sum()
                dictsubFunct['nullsurvived'] = round((dfresults[ dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar'] ].Survived.count())*avgSurvivalRate,0)
                dictsubFunct['surviveRate'] = round(dfresults[ dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar'] ].Survived.mean(),4)
                dictsubFunct['dead'] = dfresults[ dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar'] ].Survived.count() - dfresults[ dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar'] ].Survived.sum()
                dictsubFunct['total'] = dfresults[ dfresults[dictsubFunct['MTtestPar']]==dictsubFunct['testValuePar'] ].Survived.count()
            
            print(dictsubFunct, file=f)
 
            c_dfsubFunct = c_dfsubFunct.append(dictsubFunct, ignore_index=True)
                   
    # sex: male, female, pclass: 1,2,3 into male-1, male-2, male-3, female-1, etc 
    # each array should have - testNum, testArr, testCondition
    return c_dfsubFunct
    
def powerset(iterable):
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    # create all combinations of the iterable
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def main():
    
    # define variables
    tests = {1:'Sex', 2:'Embarked', 3:'farerange', 4:'agerange', 5:'familysize', 6:'Pclass'}
    parentTest = {'majorTest':'', 'testValue':[]}
    childTest = {'majorTest':'', 'testValue':[]}
    resDtypes = {'PassengerId':np.int32, 'Survived':np.int32, 'Pclass':str,'Name':str,'Sex':str, 'Age':str, 'SibSp':np.int32,'Parch':np.int32,'Ticket':str,'Fare':np.float,'Cabin':str,'Embarked':str,'Title':str,'Deck':str,'familysize':str,'Fare_Per_Person':np.float,'farerange':str,'agerange':str}
    dfresults = pan.read_csv(titanDir+iResultsfile, dtype=resDtypes)
    dffactors = pan.read_csv(titanDir+iFactorsfile, index_col=0)
    dftest=pan.DataFrame(columns=['testNum', 'testStr', 'pValue', 'correl'])
    dimTest = {'testNum':0, 'testStr': '', 'pValue':.000, 'correl':.000}
    dfsubFunct = pan.DataFrame(columns=['testNum', 'description', 'survived', 'dead', 'total','surviveRate', 'nullsurvived', 'condition','MTtestPar', 'testValuePar','MTtestChild', 'testValueChild'])
    testNum = 0
    subFuncNum = 0
    f = open(titanDir+'eval_test_multi.txt','w')     # output to go to text file
    
    # calculate average survival rate 
    avgSurvivalRate = calulate_std_survival_rate(dfresults)
    print('The average survival rate is: ' + str(avgSurvivalRate), file=f)

    # create all of the combinations
    testers = list(powerset(tests))
    
    # gather the base tests i.e. sex-pclass, etc
    for t in testers:
        passNum = 1
        # eliminate tests greateer than 2 or ba
        if (len(t) > 0) & (len(t) < 3):        # GO BACK TO ZERO         
            dimTest['testStr'] = ''
            dimTest['testArr'] = ()
            passNum = 1
            for i in range(len(t)):
                dimTest['testStr'] += tests[t[i]] + '-'
                dimTest['testArr']=np.append(dimTest['testArr'],tests[t[i]])
                if passNum == 1:
                    parentTest['testValue'] = dffactors.loc[tests[t[i]],'ranger'].values
                    parentTest['majorTest'] = tests[t[i]]
                    passNum += 1
                else:
                    childTest['testValue'] = dffactors.loc[tests[t[i]],'ranger'].values
                    childTest['majorTest'] = tests[t[i]]
                # go get the values based on the major string - place into the appropriate array (pass 1 vs pass 2) - parentTest & childTest
                # increment array
            dimTest['testStr']=dimTest['testStr'][0:len(dimTest['testStr'])-1]
            dimTest['testNum']=testNum
            dimTest['numArr']=t

            dfsubFunct = dfsubFunct.append(create_match_tests(parentTest, childTest, testNum, dimTest['testStr'],dfresults, avgSurvivalRate,f), ignore_index=True)
            
            testNum +=1
            dftest = dftest.append(dimTest, ignore_index=True)

        parentTest = {'majorTest':'', 'testValue':[]}
        childTest = {'majorTest':'', 'testValue':[]}
    
    dfsubFunct = dfsubFunct.set_index('testNum')
    for index, testRec in dftest.iterrows():
        # grab the values - place into arrays - survived and nullsurvived
        realHy = []
        nullHy = []
        realHy = dfsubFunct.loc[testRec.testNum,'survived']
        nullHy = dfsubFunct.loc[testRec.testNum,'nullsurvived']
        # set the two stat values equal to the results from the function
        dftest.loc[index,'correl'], dftest.loc[index,'pValue'] = stats.pearsonr(realHy, nullHy)

    print(dfsubFunct, file=f)
    print(dftest, file=f)
    
    dftest.to_csv(titanDir+"dftest.csv",index=False)
    dfsubFunct.to_csv(titanDir+"dfsubFunct.csv",index=False)
    # dictsubFunct = {'identifer':0, 'testNum':0, 'description':'', 'survived':0, 'dead':0, 'total':0,'surviveRate':.000}

main()
quit()