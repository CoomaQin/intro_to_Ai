from collections import deque
from queue import PriorityQueue


def BFS(m):
    searched = []
    neighbor = []
    fringe = deque()
    dim = len(m)
    fringe.append([0, 0])
    while fringe:
        tmp = fringe.popleft()
        if tmp not in searched:
            searched.append(tmp)
            if m[tmp[0], tmp[1]] == 4:
                # print('111')
                break
            update(neighbor, tmp, dim)
            for cell in neighbor:
                if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                    fringe.append(cell)
            m[tmp[0], tmp[1]] = 1
    return m


def DFS(m):
    searched = []
    neighbor = []
    fringe = []
    success = False
    dim = len(m)
    fringe.append([0, 0])
    while fringe:
        tmp = fringe.pop()
        if tmp not in searched:
            searched.append(tmp)
            if m[tmp[0], tmp[1]] == 4:
                success = True
                break
            update(neighbor, tmp, dim)
            for cell in neighbor:
                if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                    fringe.append(cell)
            m[tmp[0], tmp[1]] = 1
    return m, success


def update(neighbor, index, size):
    if index[1] != size - 1:
        neighbor.append([index[0], index[1] + 1])
    if index[0] != size - 1:
        neighbor.append([index[0] + 1, index[1]])
    # if index[0] != 0:
    #     neighbor.append([index[0] - 1, index[1]])
    # if index[1] != 0:
    #     neighbor.append([index[0], index[1] - 1])


def AStar(m):
    neighbor = []
    fringe = PriorityQueue()
    g = {"[0, 0]": 0}
    priority = 0
    success = False
    dim = len(m)
    fringe.put(heuristic(([0, 0], [dim, dim])), [0, 0])
    while fringe:
        tmp = fringe.get()
        if m[tmp[0], tmp[1]] == 4:
            success = True
            break
        update(neighbor, tmp, dim)
        for cell in neighbor:
            if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                g.update({str(cell): g[str(tmp)] + 1})
                priority = g[str(cell)] + heuristic(cell, [dim, dim])
                fringe.put(priority, cell)
        m[tmp[0], tmp[1]] = 1
    return m, success


def heuristic(start_idx, goal_idx):
    return goal_idx[0] - start_idx[0] + goal_idx[1] - start_idx[0]
