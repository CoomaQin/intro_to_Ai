import random
import numpy as np


class Cell:
    """
    a cell represents  an element in the board
    """
    def __init__(self, pos, is_mine, status='covered', neighbors=None):
        if neighbors is None:
            # number 9 represents uncovered cell
            neighbors = [9, 9, 9, 9, 9, 9, 9, 9]
        # position
        self.pos = pos
        # status contains covered, uncovered or marked
        self.status = status
        # its nearest 8 cells
        self.neighbors = neighbors
        self.is_mine = is_mine


class Board:
    def __init__(self, dim, mine_num):
        # assume board is in the shape of square
        self.dim = dim
        if mine_num > dim ** 2:
            raise ValueError('too many mines.')
        self.mine_num = mine_num
        # shore cells are still covered
        self.covered_list = np.linspace(0, dim ** 2 - 1, num=dim ** 2, dtype=int).tolist()
        cell_matrix = []
        mine = random.sample(range(0, dim ** 2 - 1), mine_num)
        # shore cells contains a mine
        mine_list = []
        for i in range(dim):
            for j in range(dim):
                if i * dim + j in mine:
                    mine_list.append([i, j])
        self.mine_list = mine_list
        # cell_matrix is a 2D list of object cell
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
        self.boom = 0
        # value_matrix is shore values and marks of cells, just like the Window mine sweeper game.
        # 10 represents marked, 9 represents covered
        self.value_matrix = 9 * np.ones([dim, dim], dtype=int)

    def query(self, idx):
        """
        use it to query a cell
        :param idx: the position of the cell
        :return:
        """
        cell = self.cell_matrix[idx[0]][idx[1]]
        if cell.status == 'uncovered':
            raise ValueError('cell ' + str(idx) + ' has been queried')
        if cell.is_mine:
            self.boom += 1
            self.mark(idx)
        else:
            cell.status = 'uncovered'
            value = 0
            for elem in self.mine_list:
                if abs(elem[0] - idx[0]) <= 1 and abs(elem[1] - idx[1]) <= 1:
                    value += 1
            self.value_matrix[idx[0], idx[1]] = value
            self.covered_list.remove(idx[0] * self.dim + idx[1])

    def mark(self, idx):
        """
        use it to mark a cell
        :param idx: the position of the cell
        :return:
        """
        if self.cell_matrix[idx[0]][idx[1]] == 'uncovered':
            raise ValueError('cell ' + str(idx) + ' has been queried')
        self.cell_matrix[idx[0]][idx[1]].status = 'marked'
        self.value_matrix[idx[0], idx[1]] = 10
        self.covered_list.remove(idx[0] * self.dim + idx[1])

    def is_gameover(self):
        m = self.value_matrix
        if len(m[m == 9]) == 0:
            return True
        else:
            return False


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
            if board.cell_matrix[i][j - 1].status == 'covered':
                neighbors.append([i, j - 1])
    if i != 0:
        if board.cell_matrix[i - 1][j].status == 'covered':
            neighbors.append([i - 1, j])
        if j != 0:
            if board.cell_matrix[i - 1][j - 1].status == 'covered':
                neighbors.append([i - 1, j - 1])
        if j != size - 1:
            if board.cell_matrix[i - 1][j + 1].status == 'covered':
                neighbors.append([i - 1, j + 1])
    return neighbors


def randow_select(board):
    """
    randomly select a cell from covered cells
    :param board:
    :return:
    """
    r = random.choice(board.covered_list)
    idx = [r // board.dim, r % board.dim]
    return idx


def is_all_safe(idx, board):
    """
    determine if all neighbors of one cell are safe
    :param idx:
    :param board:
    :return:
    """
    if board.value_matrix[idx[0], idx[1]] == 0:
        all_safe = True
    else:
        all_safe = False
    return all_safe


def is_all_mine(idx, board):
    """
    determine if all neighbors of one cell are mines
    :param idx:
    :param board:
    :return:
    """
    unmarked_list = unmarked_neighbors(idx, board)
    if board.value_matrix[idx[0], idx[1]] == len(unmarked_list):
        all_mine = True
    else:
        all_mine = False
    return all_mine
