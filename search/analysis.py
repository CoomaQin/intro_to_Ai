from search_algs import DFS, BFS, AStar, BIBFS
# from grid_draw import draw_grid
from local_search import genetic
from matrix import generate_maze
import networkx as nx
import numpy as np


def maze_solvability(p, dim, num):
    solvable_count = 0
    for i in range(num):
        mz = generate_maze(p, dim)
        _, res = DFS(mz)
        if res:
            solvable_count += 1
    return solvable_count / num


def maze_short_path(maze_matrix):
    index = np.argwhere(maze_matrix == 1)
    N = len(index)
    aj_matrix = np.zeros([N, N])
    for i in range(N):
        for j in range(i):
            if index[i][0] - index[j][0] == 1 or index[i][1] - index[j][1] == 1:
                aj_matrix[i, j] = 1
                aj_matrix[j, i] = 1
    G = nx.from_numpy_matrix(aj_matrix, create_using=nx.DiGraph())
    return nx.shortest_path(G, source=0, target=N - 1)[-1] + 1


# p = genetic(30, 0.1, 100, 20, DFS, 0.1)
# for i in p:
#     print(i.hardness)

mz = generate_maze(0.2, 10)
# mz, _ = AStar(mz, 'manhattan')
# mz, _ = BFS(mz)
# mz, _ = DFS(mz)
# mz, _ = BIBFS(mz)
mz, _ = AStar(mz, 'euclid')
print(len(mz[mz == 1]))
print(mz)
# print(maze_short_path(mz))
# print(maze_solvability(0.3, 100, 100))
