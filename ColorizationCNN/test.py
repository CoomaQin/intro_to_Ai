import numpy as np

z = np.zeros([5, 6])
z[2, 2] = 1
print(z)
print(z[:, 0:6:2])
