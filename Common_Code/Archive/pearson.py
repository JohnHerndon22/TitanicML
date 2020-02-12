#pearson.py

import numpy as np
from scipy import stats
a = np.array([0, 0, 0, 1.5, 1, 1, 1])
b = np.arange(7)
print(stats.pearsonr(a, b))