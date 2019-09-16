import numpy as np


def generate_maze(p, dim):
    '''
    :param p: the probability of cells being blocked. 0 < p < 1
    :param dim: the size of the maze is dim * dim
    :return: a dim * dim matrix
    '''
    maze = np.zeros([dim, dim], dtype=int)
    for i in range(dim - 1):
        for j in range(dim - 1):
            r = np.random.random_sample()
            if (r < p):
                maze[i, j] = 2
            else:
                maze[i, j] = 0
    # the start cell and the end cell
    maze[0, 0] = 0
    maze[dim-1, dim-1] = 0
    return maze
