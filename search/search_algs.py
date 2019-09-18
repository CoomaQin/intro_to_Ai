from collections import deque

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


