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
            self.prob_list[pos[0] * self.dim + pos[1]] = 1 - sum(self.prob_list) + self.prob_list[
                pos[0] * self.dim + pos[1]]
            # a simpler approach to update Tij in pos since /sum_i P(T in cell i) = 1
            cellpos.T_ij = self.prob_list[pos[0] * self.dim + pos[1]]
        return False


def play_rule1(num):
    b = board(num)
    count = 0
    print(b.landscape_matrix)
    print(b.target_pos)
    success = False
    while not success:
        pos_idx = b.prob_list.index(max(b.prob_list))
        pos = [pos_idx // num, pos_idx % num]
        print("search: ", pos_idx)
        # print(b.prob_list)
        success = b.observe(pos)
        count += 1
    print("find treasure with " + str(count) + " steps")


def play_rule2(num):
    b = board(num)
    count = 0
    print(b.landscape_matrix)
    print(b.target_pos)
    success = False
    while not success:
        pos_idx = 0
        prob = 0
        for idx in range(len(b.prob_list)):
            prob_tmp = b.prob_list[idx] * b.cell_matrix[idx // num][idx % num].prob_failure
            if prob_tmp > prob:
                prob = prob_tmp
                pos_idx = idx
        pos = [pos_idx // num, pos_idx % num]
        print("search:", pos_idx)
        success = b.observe(pos)
        count += 1
    print("find treasure with " + str(count) + " steps")

def play_rule1Agent(num):
    b = board(num)
    count = 0
    loc = [0, 0]
    index = 0
    locUtility = 0
    bestCellUtility = 0
    UtilitySucc = 1000
    UtilityFail = 5
    eVal = 0
    Discount = .65
    print(b.landscape_matrix)
    print(b.target_pos)
    success = False
    while not success:
        pos_idx = b.prob_list.index(max(b.prob_list))
        pos = [pos_idx // num, pos_idx % num]
        if(loc[0] == pos[0] and loc[1] == pos[1]):
            print("search: ", pos_idx)
            # print(b.prob_list)
            success = b.observe(pos)
        else :
            locUtility = b.prob_list[loc[0] * b.dim + loc[1]] * UtilitySucc + (1 - b.prob_list[loc[0] * b.dim + loc[1]] + b.prob_list[loc[0] * b.dim + loc[1]] * b.cell_matrix[loc[0]][loc[1]].prob_failure) * UtilityFail
            for i in range(abs(loc[0] - pos[0])) :
                if(loc[0] > pos[0]):
                    index = i * -1
                else:
                    index = i
                bestCellUtility += (Discount ** abs(index)) * (b.prob_list[(loc[0] + index) * b.dim + loc[1]] * UtilitySucc + (1 - b.prob_list[(loc[0] + index) * b.dim + loc[1]] + b.prob_list[(loc[0] + index) * b.dim + loc[1]] * b.cell_matrix[loc[0] + index][loc[1]].prob_failure) * UtilityFail)

            for j in range(abs(loc[1] - pos[1])):
                if (loc[1] > pos[1]):
                    index = j * -1
                else:
                    index = j
                bestCellUtility += (Discount ** (abs(index) + abs(loc[0] - pos[0]))) * (b.prob_list[pos[0] * b.dim + (loc[1] + index)] * UtilitySucc + (1 - b.prob_list[pos[0] * b.dim + (loc[1] + index)] + b.prob_list[pos[0] * b.dim + (loc[1] + index)] * b.cell_matrix[pos[0]][loc[1] + index].prob_failure) * UtilityFail)

            if(locUtility > bestCellUtility):
                success = b.observe(pos)
            else:
                if(loc[0] > pos[0]):
                    loc[0] -= 1
                elif(loc[0] < pos[0]):
                    loc[0] += 1
                elif (loc[1] > pos[1]):
                    loc[1] -= 1
                elif (loc[1] < pos[1]):
                    loc[1] += 1
        count += 1
    print("find treasure with " + str(count) + " steps")