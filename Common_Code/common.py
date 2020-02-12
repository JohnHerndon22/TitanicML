#common.py
import numpy as np
import os
import sys
import statistics as st
import matplotlib.pyplot as plt
from datetime import datetime 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
# from sklearn import preprocessing, neighbors
# from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, ShuffleSplit

titanDir =  '/users/johncyclist22/documents/Data/titanic/'
trainfile = 'train.csv'
testfile = 'test.csv'
iFactorsfile = 'factors.csv'
iResultsfile = 'results.csv'
iOutputfile = 'stats.txt'
male = 'male'
female = 'female'
titleList=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
dimension = ''
ranger = ''
survivalRate = .000
today = datetime.today()
processdate = str(today.year) + "-" + str(today.month).zfill(2) + "-" + str(today.day).zfill(2)

tests = {1:'Sex', 2:'Embarked', 3:'farerange', 4:'agerange', 5:'familysize', 6:'Pclass'}
agebins = [1,10,20,30,40,50,60,70,80,90,np.inf]
agelabels = ['1-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90+']
farebins = [0,5,10,15,20,25,30,np.inf]
farelabels = ['0-5','5-9','10-14','15-19','20-24','30-34','35+']
fsbins = [1,2,3,4,5,6,7,8,9,np.inf]
fslabels = ['1','2','3','4','5','6','7','8','9+']
fare_filler = 32                
age_filler = 29.6
classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, oob_score=True, n_jobs=-1),
        # "KNeighbors":KNeighborsClassifier(3),
        "GaussianProcess":GaussianProcessClassifier(warm_start=True),
        # "DecisionTree":DecisionTreeClassifier(max_depth=5),
        "MLP":MLPClassifier(max_iter=1750),
        "GaussianNB":GaussianNB(),
        # "AdaBoost":AdaBoostClassifier(),
        "QDA":QuadraticDiscriminantAnalysis(),
        "GradientBoosting":GradientBoostingClassifier(),
        # "ExtraTrees":ExtraTreesClassifier(),
        "LogisticRegression":LogisticRegression(max_iter=250),
        "LinearDiscriminantAnalysis":LinearDiscriminantAnalysis()
    }
# "SVM":SVC(kernel="linear", C=0.025),
        

def compute_family_size(df):
    # Creating new familysize column
    df['familysize']=df['SibSp']+df['Parch']+1 # the one if for the self
    df['familysize'] = df['familysize'].astype('str')
    
    return df

def determine_wealth(fare):
    if fare > 70:
        return 1
    else:
        return 0

def extract_title(Name, sex):
    title =  Name[Name.find(', ')+2:-(len(Name) - Name.find('. ', Name.find(', ')+2))]
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['Countess', 'Mme', 'the Countess','Lady', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if sex=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title   
        
def convert_embarked_to_int(embarked):
    if embarked == "S": 
        Nembarked = 1
    elif embarked == "C":
        Nembarked = 2
    else:
        Nembarked = 3
    return Nembarked

def convert_sex_to_int(Sex):
    if Sex == "male": 
        NSex = 1
    else:
        NSex = 0
    return NSex
