import numpy as np


def generate_maze(p, dim):
    """
    :param p: the probability of cells being blocked. 0 < p < 1
    :param dim: the size of the maze is dim * dim
    :return: a dim * dim matrix
    """
    maze = np.zeros([dim, dim], dtype=int)
    for i in range(dim - 1):
        for j in range(dim - 1):
            rv = np.random.random_sample()
            if rv < p:
                maze[i, j] = 2
            else:
                maze[i, j] = 0
    # the start cell is empty 0
    maze[0, 0] = 0
    # the end cell is marked 4
    maze[dim - 1, dim - 1] = 4
    return maze


def generate_fire_maze(p, dim):
    """
    :param p: the probability of cells being blocked. 0 < p < 1
    :param dim: the size of the maze is dim * dim
    :return: a dim * dim matrix
    """
    maze = np.zeros([dim, dim], dtype=int)
    for i in range(dim - 1):
        for j in range(dim - 1):
            rv = np.random.random_sample()
            if rv < p:
                maze[i, j] = 2
            else:
                maze[i, j] = 0
    # the start cell is empty 0
    maze[0, 0] = 0
    # the end cell is marked 4
    maze[dim - 1, dim - 1] = 4
    # the right left corner is on fire 3
    maze[0, dim - 1] = 3
    return maze


def fire_maze_update(p_fire, m):
    """
    :param p_fire: If a free cell has k burning neighbors, it will be on fire in the next time step with probability 1 − (1 − q)^k
    :param m: maze matrix
    :return: maze matrix
    """
    tmp = {}
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i, j] == 0 or m[i, j] == 1:
                # the number of neighbors on fire
                neighbors = []
                if i != len(m) - 1:
                    neighbors.append(m[i + 1, j])
                    if j != len(m) - 1:
                        neighbors.append(m[i + 1, j + 1])
                        neighbors.append(m[i, j + 1])
                    if j != 0:
                        neighbors.append(m[i + 1, j - 1])
                if i != 0:
                    neighbors.append(m[i - 1, j])
                    if j != 0:
                        neighbors.append(m[i - 1, j - 1])
                        neighbors.append(m[i, j - 1])
                    if j != len(m) - 1:
                        neighbors.append(m[i - 1, j + 1])
                k = 0
                for n in neighbors:
                    if n == 3:
                        k += 1
                tmp.update({(i, j): k})
    for key in tmp:
        rv = np.random.sample()
        if rv < 1 - (1 - p_fire) ** tmp[key]:
            m[key[0], key[1]] = 3
    return m
