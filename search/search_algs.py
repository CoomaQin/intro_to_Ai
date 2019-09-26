from collections import deque
from queue import PriorityQueue
import numpy as np
from matrix import generate_maze
from tqdm import tqdm
import copy


# BFS Search
def BFS(m):
    searched = []
    neighbor = []
    fringe = deque()
    success = False
    dim = len(m)
    fringe.append([0, 0])
    while fringe:
        # treat fringe as a queue
        tmp = fringe.popleft()
        if tmp not in searched:
            searched.append(tmp)
            if m[tmp[0], tmp[1]] == 4:
                success = True
                break
            neighbor = updateDown(tmp, dim)
            for cell in neighbor:
                if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                    fringe.append(cell)
            m[tmp[0], tmp[1]] = 1
    return m, success


# BFS both directions Search
def BIBFS(m):
    searchedA = []
    searchedB = []
    fringeA = deque()
    fringeB = deque()
    success = False
    dim = len(m)
    fringeA.append([0, 0])
    fringeB.append([dim - 1, dim - 1])
    # while we can move from both sides
    while len(fringeA) > 0 and len(fringeB) > 0:
        # treat fringe as a queue and get element from forw and backward
        tmpA = fringeA.popleft()
        tmpB = fringeB.popleft()
        if tmpA not in searchedA:
            # if we stumble upon the an element that's not in A but is marked (by B) we connected
            if m[tmpA[0], tmpA[1]] == 1:
                success = True
                break
            searchedA.append(tmpA)
            # add adjacent cells for forward
            neighborA = updateDown(tmpA, dim)
            for cell in neighborA:
                if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                    fringeA.append(cell)
            # marked visited
            m[tmpA[0], tmpA[1]] = 1
        if tmpB not in searchedB:
            # if we stumble upon the an element that's not in B but is marked (by A) we connected
            if m[tmpB[0], tmpB[1]] == 1:
                success = True
                break
            searchedB.append(tmpB)
            # add adjacent cells for backward
            neighborB = updateUp(tmpB, dim)
            for cell in neighborB:
                if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                    fringeB.append(cell)
            # marked visited
            m[tmpB[0], tmpB[1]] = 1
    return m, success


def DFS(m):
    searched = []
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
            if tmp[0] != 0 and (m[tmp[0] - 1][tmp[1]] == 0 or m[tmp[0] - 1][tmp[1]] == 4):
                fringe.append([tmp[0] - 1, tmp[1]])
            if tmp[1] != 0 and (m[tmp[0]][tmp[1] - 1] == 0 or m[tmp[0]][tmp[1] - 1] == 4):
                fringe.append([tmp[0], tmp[1] - 1])
            if tmp[1] != dim - 1 and (m[tmp[0]][tmp[1] + 1] == 0 or m[tmp[0]][tmp[1] + 1] == 4):
                fringe.append([tmp[0], tmp[1] + 1])
            if tmp[0] != dim - 1 and (m[tmp[0] + 1][tmp[1]] == 0 or m[tmp[0] + 1][tmp[1]] == 4):
                fringe.append([tmp[0] + 1, tmp[1]])
            m[tmp[0], tmp[1]] = 1
    return m, success


def updateUp(index, size):
    neighbor = []
    if index[0] != 0:
        neighbor.append([index[0] - 1, index[1]])
    if index[1] != 0:
        neighbor.append([index[0], index[1] - 1])
    if index[1] != size - 1:
        neighbor.append([index[0], index[1] + 1])
    if index[0] != size - 1:
        neighbor.append([index[0] + 1, index[1]])
    return neighbor


def updateDown(index, size):
    neighbor = []
    if index[1] != size - 1:
        neighbor.append([index[0], index[1] + 1])
    if index[0] != size - 1:
        neighbor.append([index[0] + 1, index[1]])
    if index[0] != 0:
        neighbor.append([index[0] - 1, index[1]])
    if index[1] != 0:
        neighbor.append([index[0], index[1] - 1])
    return neighbor


# experimental for a more guided dfs
def updateDFS(neighbor, index, size):
    # left top quadrant
    if ((index[0] < size / 2 - 1 and index[1] < size / 2 - 1)):
        # prioritize moving right then down
        if index[0] != size - 1:
            neighbor.append([index[0] + 1, index[1]])
        if index[1] != size - 1:
            neighbor.append([index[0], index[1] + 1])
        if index[0] != 0:
            neighbor.append([index[0] - 1, index[1]])
        if index[1] != 0:
            neighbor.append([index[0], index[1] - 1])
    # right top quadrant
    else:
        if (index[0] >= size / 2 - 1 and index[1] < size / 2 - 1):
            # prioritize moving down then right
            if index[1] != size - 1:
                neighbor.append([index[0], index[1] + 1])
            if index[0] != size - 1:
                neighbor.append([index[0] + 1, index[1]])
            if index[0] != 0:
                neighbor.append([index[0] - 1, index[1]])
            if index[1] != 0:
                neighbor.append([index[0], index[1] - 1])
        # left bottom quad
        else:
            if (index[0] < size / 2 - 1 and index[1] >= size / 2 - 1):
                # prioritize moving right then down
                if index[0] != size - 1:
                    neighbor.append([index[0] + 1, index[1]])
                if index[1] != size - 1:
                    neighbor.append([index[0], index[1] + 1])
                if index[0] != 0:
                    neighbor.append([index[0] - 1, index[1]])
                if index[1] != 0:
                    neighbor.append([index[0], index[1] - 1])
            # right bot
            else:
                if (index[0] >= size / 2 - 1 and index[1] >= size / 2 - 1):
                    # prioritize moving down then right
                    if index[0] != size - 1:
                        neighbor.append([index[0] + 1, index[1]])
                    if index[1] != size - 1:
                        neighbor.append([index[0], index[1] + 1])
                    if index[0] != 0:
                        neighbor.append([index[0] - 1, index[1]])
                    if index[1] != 0:
                        neighbor.append([index[0], index[1] - 1])


def AStar(m, param):
    fringe = PriorityQueue()
    # keep g values in a dictionary
    g = {}
    success = False
    dim = len(m)
    for i in range(dim):
        for j in range(dim):
            g.update({str([i, j]): 0})

    fringe.put((heuristic([0, 0], [dim-1, dim-1], param), [0, 0]))
    while fringe:
        _, tmp = fringe.get()
        if m[tmp[0], tmp[1]] == 4:
            success = True
            break
        neighbor = updateDown(tmp, dim)
        for cell in neighbor:
            if (m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4) and g[str(cell)] == 0:
                g.update({str(cell): g[str(tmp)] + 5})
                priority = g[str(cell)] + 5 * heuristic(cell, [dim-1, dim-1], param)
                # print(g[str(cell)])
                fringe.put((priority, cell))
        m[tmp[0], tmp[1]] = 1
    return m, success, g


def heuristic(start_idx, goal_idx, param):
    if param == 'manhattan':
        return (goal_idx[0] - start_idx[0]) + (goal_idx[1] - start_idx[1])
    if param == 'euclid':
        return np.sqrt(np.square((goal_idx[0] - start_idx[0])) + np.square((goal_idx[1] - start_idx[1])))
