import game_board as gb
import numpy as np


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
                # for elem in tmp:
                #     if elem not in sset:
                #         sset.append(elem)
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
                solvable, mines = constraint_statisfation(sset, board)
                if solvable:
                    for i in mines:
                        board.mark(i)
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


def constraint_statisfation(fringe, board):
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
        b.append(board.value_matrix[idx[0], idx[1]]-len(marked_list))
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


b = gb.Board(10, 10)
print(b.mine_list)
m = TSSP(b)
print(m)
print('the number of mines are queried: ' + str(b.boom))
