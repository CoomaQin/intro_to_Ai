import random
import numpy as np
import sympy as sp
import game_board as gb
from itertools import combinations

a = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1],
              [1, 1, 1, 1, 1]])
b = np.array([1, 1, 1, 1, 1, 1, 2])
b = b.reshape((-1, 1))
print(b)
a = np.hstack((a, b))
am = sp.Matrix(a)
g = np.array(am.rref()[0].tolist()).astype(np.int32)
print(g)
print(g[0:5])
# x = np.linalg.solve(a, b)
# print(np.array(x, dtype=int))
# c = np.zeros(2, dtype=int)
# v = np.vstack((a, c))
# print(v)
# print(np.linalg.matrix_rank(v))
# r1 = np.linalg.matrix_rank(a)
# print(r1)

# li = [0, 0, 1, 10, 1]
# li.append(2)
# print(li)
# li.pop()
# print(li)
