#linearalg.py
import numpy as np
from scipy import linalg

#create matrix
# A = np.array([[1,3,5],[2,5,1],[2,3,8]])
A = np.array([[1,3,5],[2,5,1],[2,3,8]])
print(A)

#Finding Inverse
A_inv = linalg.inv(A)
A_trans = np.array(A.transpose)
print(A_inv)
print(A_trans)
quit()
# Solving linear system
A = np.array([[1, 2], [3, 4]])
print(A)

# create vector
b = np.array([[5], [6]])
print(b)

# Solving linear system
print(np.linalg.solve(A, b))  # fast

# Finding Determinant
print(linalg.det(A))

# Computing norms 
linalg.norm(A)

# Solving linear least-squares problems and pseudo-inverses
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
c1, c2 = 5.0, 2.0
i = np.r_[1:11]
xi = 0.1*i
yi = c1*np.exp(-xi) + c2*xi
zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))
A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]]
c, resid, rank, sigma = linalg.lstsq(A, zi)
xi2 = np.r_[0.1:1.0:100j]
yi2 = c[0]*np.exp(-xi2) + c[1]*xi2
plt.plot(xi,zi,'x',xi2,yi2)
plt.axis([0,1.1,3.0,5.5])
plt.xlabel('$x_i$')
plt.title('Data fitting with linalg.lstsq')
plt.show()


