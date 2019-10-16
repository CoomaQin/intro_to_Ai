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
                for elem in tmp:
                    if elem not in sset:
                        sset.append(elem)
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
                constraint_statisfation(sset, board)
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
                for elem in tmp:
                    if elem not in sset:
                        sset.append(elem)
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
    neighbor_index = []
    var = []
    b = []
    for idx in fringe:
        covered_list = [i for i, v in enumerate(board.cell_matrix[idx[0]][idx[1]]) if v == 9]
        b.append(board.value_matrix[idx[0], idx[1]])
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
    size = len(var)
    linear_equation = np.zeros([size, size], dtype=int)
    for x in range(len(neighbor_index)):
        for idx in neighbor_index[x]:
            y = var.index(idx)
            linear_equation[x, y] = 1
    try:
        return True, np.linalg.solve(linear_equation, b)
    except np.linalg.LinAlgError:
        return False, b


b = gb.Board(10, 10)
print(b.mine_list)
m = DSSP(b)
print(m)
print('the number of mines are queried: ' + str(b.boom))
