import game_board as gb
import numpy as np
from itertools import combinations


def DSSP(board):
    # set of cells can be safely queried
    fset = []
    # fringe
    sset = []
    termination = 0
    while not board.is_gameover():
        # while termination < 10:
        if not fset:
            x = gb.randow_select(board)
            fset.append(x)
        print(sset)
        while fset:
            posx = fset.pop()
            board.query(posx)
            gb.update_neighbors(posx, board)
            tmp = gb.covered_neighbors(posx, board)
            if gb.is_all_safe(posx, board):
                for elem in tmp:
                    if elem not in fset:
                        fset.append(elem)
            else:
                sset.append(posx)
        for idx in sset:
            # marked determined mines before going back to safe set
            if gb.is_all_mine(idx, board):
                for posy in gb.covered_neighbors(idx, board):
                    board.mark(posy)
                    gb.update_neighbors(posy, board)
        for idx in sset:
            if gb.is_all_safe(idx, board):
                tmp = gb.covered_neighbors(idx, board)
                for elem in tmp:
                    if elem not in fset:
                        fset.append(elem)
                sset.remove(idx)
        # termination += 1
    return board.value_matrix


def TSSP(board):
    # set of cells can be safely queried
    fset = []
    # a temporary set, similar to but not equal to fringe
    sset = []
    csp = False
    termination = 0
    while not board.is_gameover():
        # while termination < 10:
        if not fset:
            if csp:
                csp = False
                solvable = False
                mines = []
                for combine_num in range(2, len(sset) - 1):
                    combs = combinations(sset, combine_num)
                    for li in combs:
                        solvable, mines = constraint_satisfaction(list(li), board)
                        if solvable:
                            break
                    if solvable:
                        print(mines)
                        break
                if solvable:
                    for i in mines:
                        board.mark(i)
                        sset.remove(i)
                        gb.update_neighbors(i, board)
            else:
                csp = True
                x = gb.randow_select(board)
                fset.append(x)
        while fset:
            posx = fset.pop()
            board.query(posx)
            gb.update_neighbors(posx, board)
            tmp = gb.covered_neighbors(posx, board)
            if gb.is_all_safe(posx, board):
                for elem in tmp:
                    if elem not in fset:
                        fset.append(elem)
            else:
                sset.append(posx)
        for idx in sset:
            # marked determined mines before going back to safe set
            if gb.is_all_mine(idx, board):
                for posy in gb.covered_neighbors(idx, board):
                    board.mark(posy)
                    gb.update_neighbors(posy, board)
        for idx in sset:
            if gb.is_all_safe(idx, board):
                tmp = gb.covered_neighbors(idx, board)
                for elem in tmp:
                    if elem not in fset:
                        fset.append(elem)
                sset.remove(idx)
        # termination += 1
    return board.value_matrix


def constraint_satisfaction(fringe, board):
    """
    based on a list of covered cells and the situation of their neighbors (the number of uncovered neighbors, the number
    of marked neighbor, etc), form a set of linear equations. the solution of the equation's set represent the position
    of mines (if it is solvable).
    :param fringe: a list of uncovered cells
    :param board: the mine sweeper board
    :return: (True, a list of mines) if solvable
             (False, values of cells in the fringe) otherwise
    """
    neighbor_index = []
    var = []
    b = []
    for idx in fringe:
        covered_list = [i for i, v in enumerate(board.cell_matrix[idx[0]][idx[1]].neighbors) if v == 9]
        marked_list = [i for i, v in enumerate(board.cell_matrix[idx[0]][idx[1]].neighbors) if v == 10]
        b.append(board.value_matrix[idx[0], idx[1]] - len(marked_list))
        row = []
        for elem in covered_list:
            if elem == 0:
                tmp = [idx[0] - 1, idx[1] - 1]
            elif elem == 1:
                tmp = [idx[0] - 1, idx[1]]
            elif elem == 2:
                tmp = [idx[0] - 1, idx[1] + 1]
            elif elem == 3:
                tmp = [idx[0], idx[1] - 1]
            elif elem == 4:
                tmp = [idx[0], idx[1] + 1]
            elif elem == 5:
                tmp = [idx[0] + 1, idx[1] - 1]
            elif elem == 6:
                tmp = [idx[0] + 1, idx[1]]
            else:
                tmp = [idx[0] + 1, idx[1] + 1]
            row.append(tmp)
            if tmp not in var:
                var.append(tmp)
        neighbor_index.append(row)
    sizey = len(var)
    sizex = len(b)
    linear_equation = np.zeros([sizex, sizey], dtype=int)
    for x in range(len(neighbor_index)):
        for idx in neighbor_index[x]:
            y = var.index(idx)
            linear_equation[x, y] = 1
    # print(linear_equation)
    try:
        solution = np.linalg.solve(linear_equation, b)
        mine_idx = [idx for idx, elem in enumerate(solution) if elem == 1]
        mine_list = []
        for idx in mine_idx:
            mine_list.append(var[idx])
        return True, mine_list
    except np.linalg.LinAlgError:
        return False, b


# b = gb.Board(10, 15)
# print(b.mine_list)
# m = TSSP(b)
# print(m)
# print('the number of mines are queried: ' + str(b.boom))

# test csp
# b = gb.Board(3, 0)
# b.mine_list = [[0, 2], [2, 2]]
# b.cell_matrix[0][2].is_mine = True
# b.cell_matrix[2][2].is_mine = True
# b.query([0, 0])
# gb.update_neighbors([0, 0], b)
# b.query([1, 0])
# gb.update_neighbors([1, 0], b)
# b.query([2, 0])
# gb.update_neighbors([2, 0], b)
# b.query([0, 1])
# gb.update_neighbors([0, 1], b)
# b.query([1, 1])
# gb.update_neighbors([1, 1], b)
# b.query([2, 1])
# gb.update_neighbors([2, 1], b)
# print(b.value_matrix)
# fr = [[0, 1], [1, 1], [2, 1]]
# print(constraint_satisfaction(fr, b))

# test improved guess
b = gb.Board(5, 0)
b.mine_list = [[0, 0], [0, 1], [1, 2], [1, 3], [2, 1], [3, 1], [4, 0]]
b.cell_matrix[0][0].is_mine = True
b.cell_matrix[0][1].is_mine = True
b.cell_matrix[1][2].is_mine = True
b.cell_matrix[1][3].is_mine = True
b.cell_matrix[2][1].is_mine = True
b.cell_matrix[3][1].is_mine = True
b.cell_matrix[4][0].is_mine = True

b.query([0, 2])
gb.update_neighbors([0, 2], b)
b.query([1, 0])
gb.update_neighbors([1, 0], b)
b.query([2, 0])
gb.update_neighbors([2, 0], b)
b.query([3, 0])
gb.update_neighbors([3, 0], b)
b.query([0, 3])
gb.update_neighbors([0, 3], b)
b.query([0, 4])
gb.update_neighbors([0, 4], b)

b.mark([0, 0])
gb.update_neighbors([0, 0], b)
b.mark([0, 1])
gb.update_neighbors([0, 1], b)
b.mark([1, 2])
gb.update_neighbors([1, 2], b)
b.mark([3, 1])
gb.update_neighbors([3, 1], b)
b.mark([4, 0])
gb.update_neighbors([4, 0], b)
print(b.value_matrix)
# print(b.cell_matrix[0][2].neighbors)

fr = [[0, 2], [0, 3], [0, 4], [1, 0], [2, 0], [3, 0]]
print(gb.improved_guess(fr, b))
