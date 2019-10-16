import random
import numpy as np
import game_board as gb

a = np.array([[1, 1], [1, 1]])
b = np.array([[9], [8]])
c = np.array([[0], [0]])
x = np.linalg.solve(a, b)
print(np.array(x, dtype=int))
v = np.hstack((a, b, c))
print(v)
print(np.linalg.matrix_rank(v))
r1 = np.linalg.matrix_rank(a)
print(r1)

# li = [0, 1, 0, 1, 1, 1]
# print([i for i,v in enumerate(li) if v==1])
