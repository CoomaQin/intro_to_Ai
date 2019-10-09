from matrix import generate_maze
import numpy as np
from queue import PriorityQueue
from search_algs import heuristic, updateDown
from matrix import generate_fire_maze, fire_maze_update
from tqdm import tqdm


# use A* search
def regular_search(m, param):
    neighbor = []
    fringe = PriorityQueue()
    # keep g values in a dictionary
    g = {"[0, 0]": 0}
    priority = 0
    status = 'no solution'
    dim = len(m)
    fringe.put((heuristic([0, 0], [dim, dim], param), [0, 0]))
    while fringe:
        fire_maze_update(0.3, m)
        _, tmp = fringe.get()
        if m[tmp[0], tmp[1]] == 4:
            status = 'success'
            break
        elif m[tmp[0], tmp[1]] == 3:
            # the pos you die
            m[tmp[0], tmp[1]] = 6
            status = 'dead'
            break
        neighbor = updateDown(tmp, dim)
        for cell in neighbor:
            if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                g.update({str(cell): g[str(tmp)] + 1})
                priority = g[str(cell)] + 10 * heuristic(cell, [dim - 1, dim - 1], param)
                fringe.put((priority, cell))
        m[tmp[0], tmp[1]] = 1
    return m, status


# sum = 0
# for _ in tqdm(range(15)):
#     loop = True
#     while loop:
#         maze = generate_fire_maze(0.3, 20)
#         maze, status = regular_search(maze, 'euclid')
#         if status == 'dead':
#             break
#         elif status == 'success':
#             sum += 1
#             break
# print(sum / 20)

maze = generate_fire_maze(0.3, 20)
maze, status = regular_search(maze, 'euclid')
print(maze, status)
