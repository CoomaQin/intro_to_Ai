import numpy as np
from collections import deque
from queue import PriorityQueue
# from matrix import generate_maze
import copy

# q = PriorityQueue()
#
# q.put((2, [1, 1]))
# q.put((1, [2, 2]))
# q.put((3, 'sleep'))

# m = generate_maze(0.4, 5)
# mz = copy.copy(m)
# DFS(m)
# print(m)
# print(mz)

tmp = {}
tmp.update({(1, 1): 1})
for key in tmp:
    print(key[0], key[1])