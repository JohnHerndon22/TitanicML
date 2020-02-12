for i in range(len(t)):
                testStr += tests[t[i]] + '-'
                # subfact = dffactors.loc[tests[t[i]],'ranger']
                # print(dfsubfact)
            
            dimtest['testStr']=testStr[0:len(testStr)-1] 
            dimtest['testNum']=testNum
            dftest = dftest.append(dimtest, ignore_index=True)
            testNum+=1
            print(str(testNum)+"-->"+testStr)
            functions = dffactors.loc[tests[t[i]],'ranger'].values