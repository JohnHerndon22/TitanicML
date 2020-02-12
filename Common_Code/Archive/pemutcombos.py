#permutcombos.py
import pandas as pan
import itertools
from common import *

def powerset(iterable):
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

tests = {1:'Sex', 2:'Embarked', 3:'farerange', 4:'agerange', 5:'familysize', 6:'Pclass'}
testdict = {'testNum':0, 'testStr':'', 'testArr':(), 'numArr':(), 'pValue':.000, 'correl':.000}
# dfsubFunct = pan.DataFrame()
dfsubFunct=pan.DataFrame(columns=['majorTest', 'testNum', 'subTestValue', 'condition', 'survived', 'dead', 'total','surviveRate'])
factdict = {'majorTest':'','testNum':0, 'subTestValue':'', 'condition':'', 'survived':0, 'dead':0, 'total':0,'surviveRate':.000}
dffactors = pan.read_csv(titanDir+iFactorsfile, index_col=0)
dftest=pan.DataFrame(columns=['testNum', 'testStr', 'testArr','numArr', 'pValue', 'correl'])
    
def create_match_tests(dfsubFunct):
    
    dfsubCombo=pan.DataFrame(columns=['majorTest1', 'subTestValue1', 'majorTest2', 'subTestValue2', 'subTestValue', 'condition','survived', 'dead', 'total','surviveRate'])
    combodict = {'majorTest1':'','subTest1':'','majorTest2':'','subTest2':'', 'subTest':'', 'condition':'','survived':0, 'dead':0, 'total':0,'surviveRate':.000}
    TdfsubFunct = pan.DataFrame()

    # pull all of the test number - remove duplicates

    # loop thru array by testNum
    for index,testRecord in dfsubFunct.iterrows():   #CHANGE THIS TO THE ARRAY
        print(testRecord.testNum, testRecord.majorTest, testRecord.subTestValue, testRecord.condition)
    
        # for the current test number - grab the current majortest - assign to majortest1
        combodict['majorTest1']=testRecord.majorTest

        # assign the subfunction to subfunctArr1
        combodict['subTest1'] = testRecord.subTestValue

        # loop thru the subfunctarr1 
            # assign to subTestValue1
            # if no subtestValues - assign 'blank' to subTestValue2, majorTest2
            # loop thru subfunctarr2 - assign to SubTestValue2, majorTest2
            # configure and assign the condition calculation
            # assign the dictionary to the dataframe and reset test2 to blank
        


    
    print(dfsubFunct)
    print(y)
    return 3


# print(dffactors)
# quit()

testers = list(powerset(tests))
#delete the strings longer than 2
testNum = 1
for t in testers:
    if (len(t) > 0) & (len(t) < 3):
        testdict['testStr'] = ''
        testdict['testArr'] = ()
        passNum = 1
        for i in range(len(t)):
            testdict['testStr'] += tests[t[i]] + '-'
            testdict['testArr']=np.append(testdict['testArr'],tests[t[i]])
        testdict['testStr']=testdict['testStr'][0:len(testdict['testStr'])-1]
        testdict['testNum']=testNum
        testdict['numArr']=t
        
        testNum +=1
        dftest = dftest.append(testdict, ignore_index=True)
        print(testdict['testStr'])

print(dftest)
for index,testRecord in dftest.iterrows():
    passNum = 1
    print(testRecord.numArr)
    for test in testRecord.testArr: 
        functions = dffactors.loc[test,'ranger'].values
        for funct in functions:
            factdict['subTestValue'] = funct
            factdict['majorTest'] = test
            factdict['passNum'] = passNum
            factdict['testNum'] = testRecord.testNum
            factdict['condition'] = "dfresults['"+test + "']='" + funct + "'" 
            dfsubFunct = dfsubFunct.append(factdict, ignore_index=True)
        passNum+=1

create_match_tests(dfsubFunct)

print(dfsubFunct)    



