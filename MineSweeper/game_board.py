import random
import numpy as np


class Cell:
    def __init__(self, pos, is_mine, status='covered', neighbors=None):
        if neighbors is None:
            neighbors = [9, 9, 9, 9, 9, 9, 9, 9]
        self.pos = pos
        self.status = status
        self.neighbors = neighbors
        self.is_mine = is_mine


class Board:
    def __init__(self, dim, mine_num):
        self.dim = dim
        if mine_num > dim ** 2:
            raise ValueError('too many mines.')
        self.mine_num = mine_num
        self.covered_list = range(0, dim ** 2 - 1)
        cell_matrix = []
        mine = random.sample(range(0, dim ** 2 - 1), mine_num)
        mine_list = []
        for i in range(dim):
            for j in range(dim):
                if i * dim + j in mine:
                    mine_list.append([i, j])
        self.mine_list = mine_list
        for i in range(dim):
            tmp = []
            for j in range(dim):
                if [i, j] in self.mine_list:
                    tmp_cell = Cell([i, j], True)
                else:
                    tmp_cell = Cell([i, j], False)
                tmp.append(tmp_cell)
            cell_matrix.append(tmp)
        self.cell_matrix = cell_matrix
        self.value_matrix = np.zeros([dim, dim], dtype=int)

    def query(self, idx):
        if self.cell_matrix[idx[0]][idx[1]].status == 'uncovered':
            raise ValueError('this cell has been queried')
        self.cell_matrix[idx[0]][idx[1]].status = 'uncovered'
        value = 0
        for elem in self.mine_list:
            if abs(elem[0] - idx[0]) <= 1 and abs(elem[1] - idx[1]) <= 1:
                value += 1
        self.value_matrix[idx[0], idx[1]] = value

    def mark(self, idx):
        if self.cell_matrix[idx[0]][idx[1]] == 'uncovered':
            raise ValueError('this cell has been queried')
        self.cell_matrix[idx[0]][idx[1]].status = 'marked'
        self.value_matrix[idx[0], idx[1]] = 10


def unmarked_neighbors(index, board):
    neighbors = []
    i = index[0]
    j = index[1]
    size = board.dim
    if i != size - 1:
        if board.cell_matrix[i + 1][j].status == 'covered':
            neighbors.append([i + 1, j])
        if j != size - 1:
            if board.cell_matrix[i + 1][j + 1].status == 'covered':
                neighbors.append([i + 1, j + 1])
            if board.cell_matrix[i][j + 1].status == 'covered':
                neighbors.append([i, j + 1])
        if j != 0:
            if board.cell_matrix[i + 1][j - 1].status == 'covered':
                neighbors.append([i + 1, j - 1])
    if i != 0:
        if board.cell_matrix[i - 1][j].status == 'covered':
            neighbors.append([i - 1, j])
        if j != 0:
            if board.cell_matrix[i - 1][j - 1].status == 'covered':
                neighbors.append([i - 1, j - 1])
            if board.cell_matrix[i][j - 1].status == 'covered':
                neighbors.append([i, j - 1])
        if j != size - 1:
            if board.cell_matrix[i - 1][j + 1].status == 'covered':
                neighbors.append([i - 1, j + 1])
    return neighbors


def randow_select(board):
    r = random.sample(board.covered_list, 1)
    idx = [np.floor(r / board.dim), r % board.dim]
    return idx


def is_all_safe(idx, board):
    if board.visual_matrix[idx[0], idx[1]] == 0:
        all_safe = True
    else:
        all_safe = False
    return all_safe


def is_all_mine(idx, board):
    unmarked_list = unmarked_neighbors(idx, board)
    if board.visual_matrix[idx[0], idx[1]] == len(unmarked_list):
        all_mine = True
    else:
        all_mine = False
    return all_mine


b = Board(4, 3)
b.query([0, 0])
print(b.value_matrix)
print(b.mine_list)
