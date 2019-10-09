import numpy as np
from collections import deque
from queue import PriorityQueue
# from matrix import generate_maze
import copy
import matplotlib.pyplot as plt

# dim = 100
solvability = [0.9853, 0.9795, 0.9692, 0.9597, 0.9391
    , 0.9149, 0.8872, 0.8621, 0.819, 0.7734, 0.7113, 0.6325, 0.5377, 0.4252, 0.2913, 0.1327
    , 0.0261, 0.0023, 0.0001, 0, 0, 0, 0, 0]
p = np.linspace(0.1, 0.58, num=24)
# dim = 200
solvability2 = [0.7076, 0.6376, 0.5297, 0.4888, 0.4343, 0.2833]
p2 = [0.3, 0.32, 0.34, 0.35, 0.36, 0.38]

q = [0.1, 0.2, 0.3]
sur1 = [0.8, 0.5, 0.4]
sur2 = [0.07, 0.25, 0.1]
sur3 = [0, 0, 0]
plt.plot(q, sur1, 'ro', label='p=0.1')
plt.plot(q, sur2, 'bs', label='p=0.2')
plt.plot(q, sur3, 'g^', label='p=0.3')
plt.ylabel('survival rate')
plt.xlabel('q')
plt.legend()
# plt.annotate('p_0', xy=(0.34, 0.5377), xytext=(0.4, 0.6),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )
plt.show()
