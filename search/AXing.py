#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:50:44 2019

@author: ruixigu
"""

import numpy as np


class coordinate:
    def __init__(self, x=0, y=0):
        self.x = x  # define coordinate X
        self.y = y  # define coordinate Y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False


class AStarAlgorithm:
    class Node:
        def __init__(self, point, endPoint, g=0):
            self.point = point  # current point
            self.g = g  # the value of g(n)
            self.h = abs(endPoint.x - point.x) + abs(endPoint.y - point.y)  # Manhattan Distance formula
            self.parent = None  # set parent node
            return

    def __init__(self, maze, startPoint, endPoint, blocked=2):
        self.openSet = []  # set a list to store considerable point
        self.closedSet = []  # set a list to store unconsiderable point
        self.maze = maze  # the maze we construct
        self.startPoint = startPoint  # starting point
        self.endPoint = endPoint  # destination
        self.blocked = blocked  # blocked point

    def minNodeOfF(self):
        # if len(self.openSet) == 0:
        # raise Exception('Unsuccessful path finding!')
        currentNode = self.openSet[0]  # get a node from open set
        for node in self.openSet:
            if node.g + node.h < currentNode.g + currentNode.h:  # compare the value of f
                currentNode = node  # renew the node
        return currentNode

    def closedSetPoint(self, point):  # check if the node is in the closed set
        for node in self.closedSet:
            if node.point.x != point.x:
                return False
            return True

    def openSetPoint(self, point):  # check if the node is in the open set
        # if len(self.openSet) == 0:
        # raise Exception('Unsuccessful path finding!')
        for node in self.openSet:
            if node.point != point:
                return False
            return True

    def closedSetEndPoint(self):  # check if the goal node is in the open set
        for node in self.closedSet:
            if node.point == self.endPoint:
                return node
            return False

    def search(self, minNode, XDis, YDis):
        cost = 1  # the cost from one grid to another grid
        currentPoint = coordinate(minNode.point.x + XDis, minNode.point.y + YDis)
        if self.closedSetPoint(currentPoint):  # check if this point is in the closed set
            return
        if minNode.point.x + XDis < 0 or minNode.point.x + XDis > 3 or minNode.point.y + YDis < 0 or minNode.point.y + YDis > 3:
            return None  # check if this point is out of range
        if self.maze[minNode.point.x + XDis][minNode.point.y + YDis] == self.blocked:
            return None  # check if this point is blocked
        currentNode = self.openSetPoint(currentPoint)  # check if this node is in the open set
        if not currentNode:  # if not, put this node in open set
            # if self.openSetPoint(currentPoint) == False:
            currentNode = AStarAlgorithm.Node(currentPoint, self.endPoint, g=minNode.g + cost)
            currentNode.parent = minNode  # let previous minimun node be the parent of this node
            self.openSet.append(currentNode)
            return
        else:  # if node is in the open set, check the value of g
            if minNode.g + cost < currentNode.g:  # If it is a current node that has smaller value of g
                currentNode.g = minNode.g + cost  # renew the value of g
                currentNode.parent = minNode  # renew node

    def pathFinding(self):
        startNode = AStarAlgorithm.Node(self.startPoint, self.endPoint)
        self.openSet.append(startNode)
        while len(self.openSet) > 0:
            minNode = self.minNodeOfF()
            self.closedSet.append(minNode)  # put this node into closed set, which means it won't be considered anymore
            self.openSet.remove(minNode)  # remove this node from open set
            self.search(minNode, -1, 0)  # search left
            self.search(minNode, 1, 0)  # search right
            self.search(minNode, 0, -1)  # search down
            self.search(minNode, 0, 1)  # search up
            point = self.closedSetEndPoint()  # check if end point is in the closed set
            if point:  # if it is
                pathPoint = point
                pathList = []  # set a list to store the points that have passed
                while True:
                    if pathPoint.parent:
                        pathList.append(pathPoint.point)  # put current point into the path list
                        pathPoint = pathPoint.parent  # let current point be the parent of next point
                    else:
                        return list(reversed(pathList))
            if len(self.openSet) == 0:
                raise Exception('Unsuccessful path finding')


def generate_maze(p, dim):
    '''
    :param p: the probability of cells being blocked. 0 < p < 1
    :param dim: the size of the maze is dim * dim
    :return: a dim * dim matrix
    '''
    maze = np.zeros([dim, dim], dtype=int)
    for i in range(dim - 1):
        for j in range(dim - 1):
            rv = np.random.random_sample()
            if (rv < p):
                maze[i, j] = 2
            else:
                maze[i, j] = 0
    # the start cell and the end cell
    maze[0, 0] = 0
    maze[dim - 1, dim - 1] = 4
    return maze


def main():
    maze = generate_maze(0.4, 4)
    print(maze)
    a = AStarAlgorithm(maze, coordinate(0, 0), coordinate(3, 3))
    pathList = a.pathFinding()
    for point in pathList:
        maze[point.x][point.y] = 1
    print("----------------------")
    print(maze)


if __name__ == '__main__':
    main()
