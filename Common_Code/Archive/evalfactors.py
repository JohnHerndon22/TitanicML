#pearson.py

from scipy import stats
import numpy as np
import pandas as pan
import os
import statistics as st
from common import *


dffactors = pan.read_csv(titandir+iFactorsfile, index_col=1)
factors = ['Sex', 'ticketclass', 'embarked', 'fare', 'age', 'familysize']
avgSurvivalRate = 0.348

for factor in factors:
    
    dffactorslim = dffactors.loc[dffactors['dimension'] == factor]
    # y = np.full((len(dffactorslim.index)), 0.348)

    # length = len(dffactorslim.index)    
    y = np.arange(len(dffactorslim.index))
    x = dffactorslim['survivalRate'].values
    print(factor+": ")
    print(stats.pearsonr(x, y))