# Readdata.py

import numpy as np
import pandas as pan
import os
import sys
import statistics as st
import matplotlib.pyplot as plt
from common import *


def deterAvgSurviveRate(dftrain, predColumn):
    return

def substrings_in_string(big_string, substrings):
    
    for substring in substrings:
        if substring in big_string:
            return substring
    print(big_string)
    return np.nan

def writetohist(c_dfhist, ranger, survivalRate, dimension):
    
    histdict = {'dimension': '', 'ranger': '', 'survivalRate' : .000}
    histdict['dimension'] = dimension
    histdict['ranger'] = ranger
    histdict['survivalRate'] = survivalRate
    c_dfhist = c_dfhist.append(histdict, ignore_index=True)

    return c_dfhist

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def create_df_stats(dftrain, title, dfhist):

    print('Performing calculations for: ' + title)
    print('Performing calculations for: ' + title, file=f)
    
    ## Performing calculations ##
    dftrain = dftrain.set_index('Pclass', drop=False)
    print('by class - ticket\nfirst:',file=f)
    print(round(dftrain.loc[1,'Survived'].mean(),4),file=f)
    dfhist = writetohist(dfhist, '1', round(dftrain.loc[1,'Survived'].mean(),4), 'Pclass')

    print('second: ',file=f)
    print(round(dftrain.loc[2,'Survived'].mean(),4),file=f)
    dfhist = writetohist(dfhist, '2', round(dftrain.loc[2,'Survived'].mean(),4), 'Pclass')

    print('third: ',file=f)
    print(round(dftrain.loc[3,'Survived'].mean(),4),file=f)
    dfhist = writetohist(dfhist, '3', round(dftrain.loc[3,'Survived'].mean(),4), 'Pclass')

    dftrain = dftrain.set_index('Embarked', drop=False)
    print('by class - ticket\nSouthhampton:',file=f)
    print(round(dftrain.loc['S','Survived'].mean(),4),file=f)
    dfhist = writetohist(dfhist, 'S', round(dftrain.loc['S','Survived'].mean(),4), 'Embarked')

    print('Cherbourg: ',file=f)
    print(round(dftrain.loc['C','Survived'].mean(),4),file=f)
    dfhist = writetohist(dfhist, 'C', round(dftrain.loc['C','Survived'].mean(),4), 'Embarked')

    print('Queenstown: ',file=f)
    print(round(dftrain.loc['Q','Survived'].mean(),4),file=f)
    dfhist = writetohist(dfhist, 'Q', round(dftrain.loc['Q','Survived'].mean(),4), 'Embarked')

    # fare
    # farebins = [0,5,10,15,20,25,30,np.inf]
    # farelabels = ['0-5','5-9','10-14','15-19','20-24','30-34','35+']
    dftrain = dftrain.set_index('farerange',drop=False)
    print('fare range',file=f) 
    for fare in farelabels:
        print(fare+": ",file=f)
        try:
            print(round(dftrain.loc[fare,'Survived'].mean(),4),file=f)
            dfhist = writetohist(dfhist, fare, round(dftrain.loc[fare,'Survived'].mean(),4),'farerange')
        except:
            pass

    # age range
    # agebins = [1,10,20,30,40,50,60,70,80,90,np.inf]
    # agelabels = ['1-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+']
    dftrain = dftrain.set_index('agerange',drop=False)
    print('age range',file=f) 
    for age in agelabels:
        print(age+": ",file=f)
        try:
            print(round(dftrain.loc[age,'Survived'].mean(),4),file=f)
            dfhist = writetohist(dfhist, age, round(dftrain.loc[age,'Survived'].mean(),4), 'agerange')
        except:
            dfhist = writetohist(dfhist, age, .000, 'agerange')

    # dftrain['famsizerange'] = pan.cut(dftrain['familysize'], fsbins, labels=fslabels, include_lowest=True)
    dftrain = dftrain.set_index('familysize',drop=False)
    print('family size range',file=f) 
    for fs in fsbins:
        print('fs+: ' + str(fs),file=f)
        try:
            print(round(dftrain.loc[fs,'Survived'].mean(),4),file=f)
            dfhist = writetohist(dfhist, fs, round(dftrain.loc[fs,'Survived'].mean(),4), 'familysize')
        except:
            dfhist = writetohist(dfhist, fs, .000, 'familysize')

    
    # print results and write to a file
    print("Descriptive Stats:",file=f)
    print(dftrain.describe(),file=f)
    print("Correlation of Values:",file=f)
    print(dftrain.corr(),file=f)
    print("Covariance of Values:",file=f)
    print(dftrain.cov(),file=f)


    print(dftrain.head(),file=f)


# main
f= open("prep_results.txt","w")
print('Starting processing....')
print('Starting processing....', file=f)
dfhist = pan.DataFrame()
dfresults = pan.DataFrame(columns=['PassengerId', 'Survived'])


dictresults = {'PassengerId' : 0,'survived' : 0}
os.system("clear")

dftrain = pan.read_csv(trainfile)

avg_fare = dftrain['Fare'].mean()
print('Average Fare: ' + str(avg_fare),file=f) 
avg_age = dftrain['Age'].mean()
print('Average Age: ' + str(avg_age),file=f) 


dftrain = dftrain.fillna({'Survived':0,
'Pclass':0,
'Name':'',
'Sex':'',
'Age':avg_age,
'SibSp':0,
'Parch':0,
'Ticket':'',
'Fare':avg_fare,
'Cabin':'',
'Embarked':'S'})

print('Creating columns....')
print('Creating columns....', file=f)

#Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
dftrain['Deck']=dftrain['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

# family size
# Creating new familysize column
dftrain['familysize']=dftrain['SibSp']+dftrain['Parch']+1 # the one if for the self
# create distribution of family size

# Fare per Person
dftrain['Fare_Per_Person']=dftrain['Fare']/(dftrain['familysize']+1)
dftrain['farerange'] = pan.cut(dftrain['Fare'], farebins, labels=farelabels, include_lowest=True)

# make a histogram of the ages - 0-10, 10-20, 21-30 and so on 
dftrain = dftrain.set_index('Age', drop=False)
dftrain['agerange'] = pan.cut(dftrain['Age'], agebins, labels=agelabels)

## write gender stats and create male / female datasets
dftrain = dftrain.set_index('Sex', drop=False)
dfhist = writetohist(dfhist, 'male', round(dftrain.loc['male','Survived'].mean(),4), 'Sex')
print(round(dftrain.loc['male','Survived'].mean(),4),file=f)
print('passengers: ',file=f)
print(dftrain.loc['male','Survived'].count(),file=f)
print('male: ',file=f)
dftrainmale = dftrain.loc[dftrain.index == 'male']

dfhist = writetohist(dfhist, 'female', round(dftrain.loc['female','Survived'].mean(),4), 'Sex')
print(round(dftrain.loc['female','Survived'].mean(),4),file=f)
print('passengers: ',file=f)
print(dftrain.loc['female','Survived'].count(),file=f)
dftrainfemale = dftrain.loc[dftrain.index == 'female']

create_df_stats(dftrain, "All Passengers", dfhist)
create_df_stats(dftrainmale, "Male Passengers", dfhist)
create_df_stats(dftrainfemale, "Female Passengers", dfhist)

print('Write to results file....')
print('Write to results file....', file=f)

dftrain.to_csv(iResultsfile,index=False)
dfhist.to_csv(iFactorsfile,index=False)
print('The End.', file=f)
f.close()
print('The End.')

