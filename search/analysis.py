
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from search.matrix import generate_maze
from search.search_algs import *

def maze_solvability(p, dim, num):
    solvable_count = 0
    for i in range(num):
        mz = generate_maze(p, dim)
        _, res = DFS(mz)
        if res:
            solvable_count += 1
    return solvable_count / num

#turn maze into an adjacency matrix
#then into a graph and use nx.shortest_path to get the list
#of nodes visited with a list of nodes index of the shortest path within the first list
def maze_short_path(maze_matrix):
    index = np.argwhere(maze_matrix == 1)
    N = len(index)
    aj_matrix = np.zeros([N, N], dtype=int)
    for i in range(N):
        for j in range(i):
            x = index[i][0] - index[j][0]
            y = index[i][1] - index[j][1]
            if x == 1 and y == 0:
                aj_matrix[i, j] = 1
                aj_matrix[j, i] = 1
            if x == 0 and y == 1:
                aj_matrix[i, j] = 1
                aj_matrix[j, i] = 1
    G = nx.from_numpy_matrix(aj_matrix, create_using=nx.DiGraph())
    return index, nx.shortest_path(G, source=0, target=N-1)

#turn maze into an adjacency matrix
#then into a graph and use nx.all_shortest_paths_path to get the list
#of nodes visited with a generator of node indeces which are visited and form a path to finish
def maze_short_paths(maze_matrix):
    index = np.argwhere(maze_matrix == 1)
    N = len(index)
    aj_matrix = np.zeros([N, N], dtype=int)
    for i in range(N):
        for j in range(i):
            x = index[i][0] - index[j][0]
            y = index[i][1] - index[j][1]
            if x == 1 and y == 0:
                aj_matrix[i, j] = 1
                aj_matrix[j, i] = 1
            if x == 0 and y == 1:
                aj_matrix[i, j] = 1
                aj_matrix[j, i] = 1
    G = nx.from_numpy_matrix(aj_matrix, create_using=nx.DiGraph())
    return index, nx.all_shortest_paths(G, source=0, target=N-1)

def compare_distance():
    dim = np.linspace(100, 200, num=10, dtype=int)
    print(dim)
    num_manhattan = []
    num_euclidean = []
    for d in tqdm(dim):
        success = False
        while not success:
            mz = generate_maze(0.2, d)
            mzc = copy.copy(mz)
            mz, success, _ = AStar(mz, 'manhattan')
            num_manhattan.append(len(mz[mz == 1]))
            mzc, _, _ = AStar(mzc, 'euclid')
            num_euclidean.append(len(mzc[mzc == 1]))
    plt.plot(dim, num_manhattan, color='green', label='Manhattan')
    plt.plot(dim, num_euclidean, color='red', label='Euclidean')
    plt.ylabel('The number of searched cells')
    plt.xlabel('dim')
    plt.legend()
    plt.show()


def SP_analysis():
    p = np.linspace(0.14, 0.34, num=10)
    print(p)
    avg = []
    for i in p:
        total = 0
        for _ in tqdm(range(100)):
            success = False
            while not success:
                mz = generate_maze(i, 100)
                # mz, success = DFS(mz)
                # mz, success = BFS(mz)
                mz, success, _ = AStar(mz, 'manhattan')
                # mz, success, _ = DFS(mz, 'euclid')
                # mz, success = BIBFS(mz)
                _, SP = maze_short_path(mz)
            total += len(SP)
        avg.append(total / 100)
    print(avg)
    plt.plot(p, avg)
    plt.title('BFS')
    plt.ylabel('average SP')
    plt.xlabel('p')
    plt.show()


# SP_analysis()
#mz = generate_maze(0.2, 20)
#mzc = copy.copy(mz)
#mz, success, _ = AStar(mz, 'euclid')
#mzc, _ = BIBFS(mzc)