import numpy as np
from collections import deque
from queue import PriorityQueue


q = PriorityQueue()

q.put((2, [1, 1]))
q.put((1, [2, 2]))
q.put((3, 'sleep'))

_, a = q.get()

g = {"[0, 0]": 0}

print(g[[0, 0]])

