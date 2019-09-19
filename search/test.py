import numpy as np
from collections import deque
from queue import PriorityQueue


A = np.zeros([4, 4])
A[2,2]=1
A[3,3]=1

print(np.argwhere(A==1))
