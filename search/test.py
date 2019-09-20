import numpy as np
from collections import deque
from queue import PriorityQueue
from matrix import generate_maze
from search_algs import DFS, BFS, AStar
import copy

q = PriorityQueue()

q.put((2, [1, 1]))
q.put((1, [2, 2]))
q.put((3, 'sleep'))

m = generate_maze(0.4, 5)
mz = copy.copy(m)
DFS(m)
print(m)
print(mz)

# li = np.where(m == 2)
# print(m)
# print(li)
# print([li[0][0], li[1][0]])
