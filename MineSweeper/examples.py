import game_board as gb
from draw import draw_board
from alg import TSMP, constraint_satisfaction_gauss, constraint_satisfaction

b = gb.Board(10, 20)
print(b.mine_list)
m = TSMP(b, True, True)
print(m)
print('the number of mines are queried: ' + str(b.boom))

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
# draw_board(b.value_matrix)
# fr = [[0, 1], [1, 1], [2, 1]]
# print(constraint_satisfaction_gauss(fr, b))

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
b.mine_left = 7

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
draw_board(b.value_matrix)
fr = [[0, 2], [0, 3], [0, 4], [1, 0], [2, 0], [3, 0]]
print(gb.improved_guess(fr, b))
