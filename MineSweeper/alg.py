import game_board as gb
import numpy as np
from itertools import combinations
import sympy as sp


def DSSP(board):
    # set of cells can be safely queried
    fset = []
    # fringe
    sset = []
    while not board.is_gameover():
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
    return board.value_matrix


def TSMP(board, csp_enable=False, improved_guess_enable=False):
    # set of cells can be safely queried
    fset = []
    # a temporary set, similar to but not equal to fringe
    sset = []
    csp = False
    while not board.is_gameover():
        if not fset:
            if csp and csp_enable:
                csp = False
                solvable = False
                mines = []
                # for combine_num in range(2, len(sset) - 1):
                #     combs = combinations(sset, combine_num)
                #     for li in combs:
                #         solvable, mines = constraint_satisfaction(list(li), board)
                #         if solvable:
                #             print(1)
                #             break
                #     if solvable:
                #         print(mines)
                #         break
                solvable, mines = constraint_satisfaction_gauss(sset, board)
                if solvable:
                    for i in mines:
                        board.mark(i)
                        sset.remove(i)
                        gb.update_neighbors(i, board)
            else:
                csp = True
                if improved_guess_enable and board.mine_left <= board.mine_num / 4:
                    success, x = gb.improved_guess(sset, board)
                    if not success:
                        x = [gb.randow_select(board)]
                else:
                    x = [gb.randow_select(board)]
                fset.extend(x)
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
    return board.value_matrix


def TSMPUncertain(p, board, csp_enable=False, improved_guess_enable=False):
    # set of cells can be safely queried
    fset = []
    # a temporary set, similar to but not equal to fringe
    sset = []
    csp = False
    while not board.is_gameover():
        if not fset:
            if len(sset) == 0:
                x = [gb.randow_select(board)]
                fset.extend(x)
            elif csp and csp_enable:
                csp = False
                solvable = False
                mines = []
                # for combine_num in range(2, len(sset) - 1):
                #     combs = combinations(sset, combine_num)
                #     for li in combs:
                #         solvable, mines = constraint_satisfaction(list(li), board)
                #         if solvable:
                #             print(1)
                #             break
                #     if solvable:
                #         print(mines)
                #         break
                solvable, mines = constraint_satisfaction_gauss(sset, board)
                if solvable:
                    for i in mines:
                        board.mark(i)
                        if(sset.__contains__(i)):
                            sset.remove(i)
                        if(board.cell_matrix[i[0]][i[1]].status != 'not a mine'):
                            gb.update_neighbors(i, board)
            else:
                csp = True
                if improved_guess_enable and board.mine_left <= board.mine_num / 4:
                    success, x = gb.improved_guess(sset, board)
                    if not success:
                        x = [gb.randow_select(board)]
                else:
                    x = [gb.randow_select(board)]
                fset.extend(x)
        while fset:
            posx = fset.pop()
            board.queryUncertain(posx, p)
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
        for idx in sset:
            if board.cell_matrix[idx[0]][idx[1]]:
                sset.remove(idx)
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


def constraint_satisfaction_gauss(fringe, board):
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
    b = np.array(b).reshape((-1, 1))
    agumented_matrix = np.hstack((linear_equation, b))
    am = sp.Matrix(agumented_matrix)
    echelon = np.array(am.rref()[0].tolist()).astype(np.int32)
    coefficient_matrix = echelon[:, 0:sizey]
    solution = echelon[:, sizey]
    x = []
    for i in range(0, sizex):
        if len(np.where(coefficient_matrix[i, :] > 0)[0]) == 1:
            x.append(solution[i])
    if x:
        mine_idx = [idx for idx, elem in enumerate(x) if elem == 1]
        mine_list = []
        for idx in mine_idx:
            mine_list.append(var[idx])
        return True, mine_list
    else:
        return False, b
