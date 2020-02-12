import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# dataArr = [[0, 2, 1000],[3000, 40, 3]] 
dataArr = [[1, 2, 4, 30, 40, 13], [10, 2, 45, 15, 40, 13]]
imp.fit(dataArr)

# X = [[np.nan, 2 ,6],[1000, np.nan, 6]]
X = [[np.nan, 2 ,6, 21, np.nan, 6], [1, 2, 45, 15, 40, 13]]
print(imp.transform(X))