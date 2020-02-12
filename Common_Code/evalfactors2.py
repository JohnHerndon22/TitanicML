#evalfactors2.py
#evaluates factors at a high level - one factor only - determines and stores the mean value by factor

from scipy import stats
import numpy as np
import pandas as pan
import os
import statistics as st
from common import *

def pullresultset(dfresults, tests):
    # send the result dataframe with a dimension list - array - return dataframe subset
    return

def calulate_std_survival_rate(dfresults):
    # compute the standard rate
    return round((dfresults.loc[dfresults.index,'Survived'].sum()) / (dfresults.loc[dfresults.index,'Survived'].count()),3)

# open files and define variables not found in common
dffactors = pan.read_csv(titanDir+iFactorsfile, index_col=1)
dfresults = pan.read_csv(titanDir+iResultsfile)                 # this came from the preparatory readdata.py
factors = {1:'Sex', 2:'Embarked', 3:'farerange', 4:'agerange', 5:'familysize', 6:'Pclass'}
avgSurvivalRate = calulate_std_survival_rate(dfresults)
f = open(titanDir+'eval_factors.txt','w')     # output to go to text file

for key in factors:
    
    dfresults = dfresults.set_index(factors[key],drop=False)
    dffactorslim = dffactors.loc[dffactors['dimension'] == factors[key]]
    # mean=[]
    meandict={'dim':'','meanValue':.000, 'totalSurvived': 0, 'totalDead': 0, 'totalPopulation': 0}   
    dfmean = pan.DataFrame()
    x=[]
    y=[]
    for var in dffactorslim.iterrows():
        if var[0].isdigit() == True:
            var=list(var)
            var[0]=int(var[0])
            var=tuple(var)
        try:
            # null hytpothesis
            y = np.append(y,round((dfresults.loc[var[0],'Survived'].count()*avgSurvivalRate),0))
            # alternative hypothesis
            x = np.append(x,dfresults.loc[var[0],'Survived'].sum())
            meandict['dim'] = var[0]
            meandict['meanValue'] = round(dfresults.loc[var[0],'Survived'].mean(),4)
            meandict['totalSurvived'] = dfresults.loc[var[0],'Survived'].sum()
            meandict['totalDead'] = dfresults.loc[var[0],'Survived'].count() - dfresults.loc[var[0],'Survived'].sum()
            meandict['totalPopulation'] = dfresults.loc[var[0],'Survived'].count()

        except:
            break
        dfmean = dfmean.append(meandict, ignore_index=True)

    # determine correlation and pValue
    correl, pValue = stats.pearsonr(x, y)
    print(factors[key]+": "+str(correl) + " ---- " + str(pValue), file=f)
    print(dfmean, file=f)

    
