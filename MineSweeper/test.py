import random
import numpy as np
import sympy as sp
import game_board as gb
from scipy.special import comb
from itertools import combinations

# a = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1],
#               [1, 1, 1, 1, 1]])
# print(np.linalg.matrix_rank(a))
# b = np.array([1, 1, 1, 1, 1, 1, 2])
# b = b.reshape((-1, 1))
# a = np.hstack((a, b))
# print(np.linalg.matrix_rank(a))
# am = sp.Matrix(a)
# g = np.array(am.rref()[0].tolist()).astype(np.int32)
# print(g)
# print(g[:, -1])

a = np.array([[1, 1, 1, 1], [1, 0, 0, 1]])
am = sp.Matrix(a)
g = np.array(am.rref()[0].tolist()).astype(np.int32)
print(g)
a = g[:,0:3]
b = g[:,3]
print(a)
for i in range(0,1):
    if len(np.where(a[i,:]>0)[0]) == 1:
        print(b[i])
# print(b)