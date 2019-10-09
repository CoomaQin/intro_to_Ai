import game_board as gb
import numpy as np


def DSSP(board):
    fset = []
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
        termination += 1
    return board.value_matrix


b = gb.Board(6, 4)
print(b.mine_list)
m = DSSP(b)
print(m)
print('the number of mines are queried: '+ str(b.boom))
