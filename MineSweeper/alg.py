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
            tmp = gb.unmarked_neighbors(posx, board)
            if gb.is_all_safe(posx, board):
                for elem in tmp:
                    if elem not in fset:
                        fset.append(elem)
            else:
                for elem in tmp:
                    if elem not in fset:
                        sset.append(elem)
        for idx in sset:
            # marked determined mines before going back to safe set
            if gb.is_all_mine(idx, board):
                for y in gb.unmarked_neighbors(idx, board):
                    board.mark(y)
        for idx in sset:
            if gb.is_all_safe(idx, board):
                tmp = gb.unmarked_neighbors(idx, board)
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
                constraint_statisfation(sset, board)
            else:
                x = gb.randow_select(board)
                fset.append(x)
        while fset:
            posx = fset.pop()
            board.query(posx)
            tmp = gb.covered_neighbors(posx, board)
            if gb.is_all_safe(posx, board):
                for elem in tmp:
                    if elem not in fset:
                        fset.append(elem)
            else:
                for elem in tmp:
                    if elem not in fset:
                        sset.append(elem)
        for idx in sset:
            # marked determined mines before going back to safe set
            if gb.is_all_mine(idx, board):
                for y in gb.covered_neighbors(idx, board):
                    board.mark(y)
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
    set = [item for item in fringe if item > 2]
    # not finish



b = gb.Board(6, 4)
print(b.mine_list)
m = DSSP(b)
print(m)
print('the number of mines are queried: ' + str(b.boom))
