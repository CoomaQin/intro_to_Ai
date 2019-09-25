from matrix import generate_maze
import numpy as np
from queue import PriorityQueue
from search_algs import heuristic, updateDown
from matrix import generate_fire_maze, fire_maze_update

# use A* search
def regular_search(m, param):
    neighbor = []
    k = 0
    fringe = PriorityQueue()
    # keep g values in a dictionary
    g = {"[0, 0]": 0}
    priority = 0
    status = 'no solution'
    dim = len(m)
    fringe.put((heuristic([0, 0], [dim, dim], param), [0, 0]))
    while fringe:
        fire_maze_update(0.3, m)
        k+=1
        _, tmp = fringe.get()
        if m[tmp[0], tmp[1]] == 4:
            status = 'success'
            break
        elif m[tmp[0], tmp[1]] == 3:
            # the pos you die
            m[tmp[0], tmp[1]] = 6
            status='dead'
            break
        neighbor = updateDown(tmp, dim)
        for cell in neighbor:
            if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                g.update({str(cell): g[str(tmp)] + 1})
                priority = g[str(cell)] + 10 * heuristic(cell, [dim - 1, dim - 1], param)
                fringe.put((priority, cell))
        m[tmp[0], tmp[1]] = 1
    print(k)
    return m, status


maze = generate_fire_maze(0.2, 15)
print(maze)
# for _ in range(3):
#     print(fire_maze_update(0.3, maze))
#     print('-------------------------')
maze, success = regular_search(maze, 'euclid')
print(maze)
print(success)
