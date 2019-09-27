import collections
from queue import PriorityQueue
from search.analysis import maze_short_path, maze_short_paths
from search.matrix import *



# BFS Search implementation will always look into down and right nodes first as that
#is where our goal is
#marks path traveled with 1s on the matrix input
#returns the marked matrix and a boolean value that evaluates if alg could reach goal
def BFS(m):
    searched = []
    neighbor = []
    fringe = collections.deque()
    success = False
    dim = len(m)
    fringe.append([0, 0])
    while fringe:
        # treat fringe as a queue
        tmp = fringe.popleft()
        if tmp not in searched:
            searched.append(tmp)
            if m[tmp[0], tmp[1]] == 4:
                # print('111')
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
    fringeA = collections.deque()
    fringeB = collections.deque()
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

#DFS search implementation that will always look into down and right nodes first as that
#is where our goal is
#marks path traveled with 1s on the matrix input
#returns the marked matrix and a boolean value that evaluates if alg could reach goal
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

#loads nodes in priority to get to top left node fastest when using a queue(BD-BFS)
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

#loads nodes in priority to get to down right node fastest when using a queue(BFS, BD-BFS)
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


# experimental for a more guided dfs as in analysis questions
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

#A* search that uses a heuristic to estimate how far each potential next node is from finish node
#sorts them in a priority queue and goes to most prioritized node first
def AStar(m, param):
    neighbor = []
    fringe = PriorityQueue()
    # keep g values in a dictionary
    g = {"[0, 0]": 0}
    priority = 0
    success = False
    dim = len(m)
    fringe.put((heuristic([0, 0], [dim, dim], param), [0, 0]))
    while fringe:
        _, tmp = fringe.get()
        if m[tmp[0], tmp[1]] == 4:
            success = True
            break
        neighbor = updateDown(tmp, dim)
        for cell in neighbor:
            if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                g.update({str(cell): g[str(tmp)] + 1})
                priority = g[str(cell)] + 10 * heuristic(cell, [dim-1, dim-1], param)
                fringe.put((priority, cell))
        m[tmp[0], tmp[1]] = 1
    return m, success

#A* version that can start and go towars custom locations
#used in fire maze to get multiple paths
#unlike regular A* this one returns a path which is a list of nodes that need to be
#visited to get to the end parameter node
def AStarSpecific(m, param, start, end, path):
    neighbor = []
    fringe = PriorityQueue()
    # keep g values in a dictionary
    g = {"[0, 0]": 0}
    priority = 0
    success = False
    dim = len(m)
    fringe.put((heuristic([start[0], start[1]], [end[0], end[0]], param), [0, 0]))
    while fringe:
        _, tmp = fringe.get()
        if m[tmp[0], tmp[1]] == 4 or m[end[0], end[1]] == m[tmp[0], tmp[1]]:
            success = True
            break
        neighbor = updateDown(tmp, dim)
        for cell in neighbor:
            if m[cell[0], cell[1]] == 0 or m[cell[0], cell[1]] == 4:
                g.update({str(cell): g[str(tmp)] + 1})
                priority = g[str(cell)] + 10 * heuristic(cell, [end[0], end[0]], param)
                fringe.put((priority, cell))
        m[tmp[0], tmp[1]] = 1
        path.append([tmp[0], tmp[1]])
    return path, success

#heuristic fucntion for A* search
#evaluates a heuristic from start index to end index, based on param
#param = manhattan => use manhattan distance as heuristic
#param = euclid => use euclid distance as heuristic
def heuristic(start_idx, goal_idx, param):
    if param == 'manhattan':
        return (goal_idx[0] - start_idx[0]) + (goal_idx[1] - start_idx[1])
    if param == 'euclid':
        return int(np.sqrt(np.square(10*(goal_idx[0] - start_idx[0])) + np.square(10*(goal_idx[1] - start_idx[1]))))

#find shortest path in a solvable maze and take it while fire spreads
#by default runs 10 times and prints out success rate, change l range to run more or fewer times
def fireMazeShortestPath():
    sucessrate=0
    q = .1
    for l in range(10):
        i=0
        j=0
        k=1
        print(l)
        m = generate_fire_maze(.35, 100)
        while not(BFS(m)[1]):
            m = generate_fire_maze(.35, 100)

        path = maze_short_path(m)
        #timestep loop
        for k in path[1]:
            #print( str(i) + ", " + str(j))
            fire_maze_update(q, m)
            i = (path[0])[k][0]
            j = (path[0])[k][1]
            if m[i][j] == 3:
                print("Dead at " + str(i) + ", " + str(j))
                break
            elif (i == len(m)-2 and j == len(m)-1) or (i == len(m)-1 and j == len(m)-2):
                print("Success")
                sucessrate += 1
                break
    print("q: ", q)
    print("success rate: ", sucessrate/20)
    sucessrate=0
    q+=.1

#same as fireshortestpath, but finds p paths and evaluates them based on
#euclid distance from top right(fire origin)
#then runs the best fit path
def fireeuclid():
    q = .2
    successRate = 0
    for h in range(10):
        i = 0
        j = 0
        p = 0
        avgOfPath = 0
        lengthOfPath = 0
        bestAvg = 0
        shortestLength = 0
        m = generate_fire_maze(.35, 100)
        while not (BFS(m)[1]):
            m = generate_fire_maze(.35, 100)

        #get all paths
        #paths[0] - list of nodes that can be visited
        #paths[1] - generator of paths that returns list of indices that corresponds
        #with paths[0] nodes that need to be visited to get to finish
        paths = maze_short_paths(m)

        #find best fit path
        for path in paths[1]:
            if(p > 15000):
                break
            p+=1
            for index in path:
                cell = (paths[0])[index]
                avgOfPath += heuristic(cell, [0, len(m)-1], 'euclid')
                lengthOfPath += 1
            #get avg euclidian distance of path
            avgOfPath = avgOfPath / lengthOfPath
            #if it is longer than this is the new best path
            if bestAvg <= avgOfPath and lengthOfPath < shortestLength and (paths[0])[path[len(path)-1]][0] == 99 and (paths[0])[path[len(path) - 1]][1] == 98:
                bestAvg = avgOfPath
                bestPath = path
                shortestLength = lengthOfPath
            elif bestAvg < avgOfPath and (paths[0])[path[len(path)-1]][0] == 99 and (paths[0])[path[len(path) - 1]][1] == 98:
                bestAvg = avgOfPath
                bestPath = path
                shortestLength = lengthOfPath
        #run through the path
        for k in bestPath:
            #fire spread
            fire_maze_update(q, m)
            #check if we are in fire
            if m[i][j] == 3:
                print("Dead at " + str(i) + ", " + str(j))
                break
            # get the next indices for the next path node
            i = (paths[0])[k][0]
            j = (paths[0])[k][1]
            #check if we moved into fire
            if m[i][j] == 3:
                print("Dead at " + str(i) + ", " + str(j))
                break
            #since we moved fire before we moved if we are adjacent to the end block we can move to it and win
            elif (i == len(m) - 2 and j == len(m) - 1) or (i == len(m) - 1 and j == len(m) - 2):
                print("Success")
                successRate += 1
                break
        print("At " + str(i) + ", " + str(j))
    #divide successrate by h
    print("success : ", successRate/10)
    print("q : ", q)
