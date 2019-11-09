import random
import numpy as np


class cell:
    def __init__(self, type, pos, prob_failure, prob, target_here=False):
        """
        :param type: the type of terrain
        :param pos:
        :param target_here:
        :param target_here: the P(Target in this Cell)
        """
        self.type = type
        self.pos = pos
        self.target_here = target_here
        self.prob_failure = prob_failure
        self.T_ij = prob


class board:
    def __init__(self, dim):
        cell_matrix = []
        self.num = dim ** 2
        self.dim = dim
        self.landscape_matrix = np.zeros([dim, dim], dtype=int)
        for i in range(dim):
            tmp = []
            for j in range(dim):
                r = random.uniform(0, 1)
                if r < 0.2:
                    tmp.append(cell("flat", [i, j], 0.1, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 1
                elif r < 0.5:
                    tmp.append(cell("hilly", [i, j], 0.3, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 2
                elif r < 0.8:
                    tmp.append(cell("forested", [i, j], 0.7, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 3
                else:
                    tmp.append(cell("caves", [i, j], 0.9, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 4
            cell_matrix.append(tmp)
        target = random.sample(range(0, dim - 1), 2)
        cell_matrix[target[0]][target[1]].target_here = True
        self.target_pos = target
        self.cell_matrix = cell_matrix
        self.prob_list = []
        for i in range(self.num):
            self.prob_list.append(1 / self.num)

    def observe(self, pos):
        r = random.uniform(0, 1)
        cellpos = self.cell_matrix[pos[0]][pos[1]]
        posij = self.cell_matrix[pos[0]][pos[1]].T_ij
        if cellpos.target_here and r < 1 - cellpos.prob_failure:
            return True
        else:
            for i in range(self.dim):
                for j in range(self.dim):
                    cellij = self.cell_matrix[i][j]
                    # P{O_t | !T_i} = (1 - prob_observation_true_notTij) + cellij.prob_failure
                    #                                                           * prob_observation_true_notTij
                    prob_observation_true_notTij = cellij.T_ij / (sum(self.prob_list) - posij)
                    # update Tij in the landscape
                    cellij.T_ij = cellij.T_ij / (cellij.T_ij + (1 - cellij.T_ij) * (
                            (1 - prob_observation_true_notTij) + cellpos.prob_failure * prob_observation_true_notTij))
            # update Tij in pos
            # cellpos.Tij = cellpos.prob_failure * cellpos.T_ij / (
            #         cellpos.prob_failure * cellpos.T_ij + (1 - cellpos.T_ij))
            # update prob_list
            for i in range(self.dim):
                for j in range(self.dim):
                    self.prob_list[self.dim * i + j] = self.cell_matrix[i][j].T_ij
            self.prob_list[pos[0] * self.dim + pos[1]] = 1 - sum(self.prob_list) + self.prob_list[pos[0] * self.dim + pos[1]]
            cellpos.T_ij = self.prob_list[pos[0] * self.dim + pos[1]]
        return False


def play(num):
    b = board(num)
    print(b.landscape_matrix)
    print(b.target_pos)
    success = False
    while not success:
    # for i in range(50):
        pos_idx = b.prob_list.index(max(b.prob_list))
        pos = [pos_idx // num, pos_idx % num]
        print(pos_idx)
        print(b.prob_list)
        success = b.observe(pos)
    print("find treasure")


play(3)
# b = board(3)
# print(b.landscape_matrix)
# print(b.target_pos)
# _ = b.observe([0, 0])
# print(b.prob_list)
# _ = b.observe([0, 1])
# print(b.prob_list)

