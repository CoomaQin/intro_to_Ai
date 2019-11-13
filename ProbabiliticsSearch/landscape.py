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
        self.obervation = []
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

    # update P(T in cell_i) with P(T in cell_i | O)
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

    # use up to t observations in each update
    def observe_subtitute(self, pos):
        r = random.uniform(0, 1)
        self.obervation.append(pos)
        cellpos = self.cell_matrix[pos[0]][pos[1]]
        if cellpos.target_here and r < 1 - cellpos.prob_failure:
            return True
        else:
            right_prob = 1
            for x in range(self.dim):
                for y in range(self.dim):
                    cellxy = self.cell_matrix[x][y]
                    right_prob *= cellxy.prob_failure ** self.obervation.count([x, y])
            for i in range(self.dim):
                for j in range(self.dim):
                    cellij = self.cell_matrix[i][j]
                    left_prob = 1 / self.dim ** 2 * cellij.prob_failure ** self.obervation.count(pos)
                    right_prob = (right_prob - cellij.prob_failure ** self.obervation.count([i, j])) / self.dim ** 2
                    cellij.T_ij = left_prob / (left_prob + right_prob)
        return False


class moving_board:
    def __init__(self, dim):
        cell_matrix = []
        self.num = dim ** 2
        self.dim = dim
        self.landscape_matrix = np.zeros([dim, dim], dtype=int)
        self.terrain = ["flat", "hilly", "forested", "caves"]
        self.count = [0, 0, 0, 0]
        for i in range(dim):
            tmp = []
            for j in range(dim):
                r = random.uniform(0, 1)
                if r < 0.2:
                    tmp.append(cell("flat", [i, j], 0.1, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 1
                    self.count[0] += 1
                elif r < 0.5:
                    tmp.append(cell("hilly", [i, j], 0.3, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 2
                    self.count[1] += 1
                elif r < 0.8:
                    tmp.append(cell("forested", [i, j], 0.7, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 3
                    self.count[2] += 1
                else:
                    tmp.append(cell("caves", [i, j], 0.9, 1 / dim ** 2))
                    self.landscape_matrix[i, j] = 4
                    self.count[3] += 1
            cell_matrix.append(tmp)
        target = random.sample(range(0, dim - 1), 2)
        cell_matrix[target[0]][target[1]].target_here = True
        self.target_pos = target
        self.cell_matrix = cell_matrix
        # Y_1
        self.tracker_report = random.choice(self.terrain)
        while self.tracker_report == self.cell_matrix[target[0]][target[1]].type:
            self.tracker_report = random.choice(self.terrain)
        # init beliefs
        self.beliefs = []
        for k in range(self.num):
            if self.cell_matrix[k // self.dim][k % self.dim].type == self.tracker_report:
                self.beliefs.append(0)
            else:
                self.beliefs.append(1 / (self.num - self.count[self.terrain.index(self.tracker_report)]))

    def get_neighbors(self, pos):
        neighbor = []
        if pos[0] != 0:
            neighbor.append([pos[0] - 1, pos[1]])
        if pos[0] != self.dim - 1:
            neighbor.append([pos[0] + 1, pos[1]])
        if pos[1] != 0:
            neighbor.append([pos[0], pos[1] - 1])
        if pos[1] != self.dim - 1:
            neighbor.append([pos[0], pos[1] + 1])
        return neighbor

    def target_move(self):
        target_neighbor = self.get_neighbors(self.target_pos)
        r = random.randint(0, len(target_neighbor) - 1)
        self.target_pos = target_neighbor[r]
        self.tracker_report = random.choice(self.terrain)
        while self.tracker_report == self.cell_matrix[self.target_pos[0]][self.target_pos[1]].type:
            self.tracker_report = random.choice(self.terrain)

    def update_belief(self):
        beta = 10
        for k in range(self.num):
            pos = [k // self.dim, k % self.dim]
            k_neighbor = self.get_neighbors(pos)
            if self.cell_matrix[k // self.dim][k % self.dim].type == self.tracker_report:
                left_belief = 0
            else:
                left_belief = beta
            right_belief = 0
            for n in k_neighbor:
                right_belief += self.beliefs[n[0] * self.dim + n[1]] / len(k_neighbor)
            self.beliefs[k] = right_belief * left_belief
        print(self.beliefs)

    def search(self, pos):
        r = random.uniform(0, 1)
        cellij = self.cell_matrix[pos[0]][pos[1]]
        if pos == self.target_pos and r < cellij.prob_failure:
            return True
        else:
            self.target_move()
            self.update_belief()
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


def play_moving(num):
    b = moving_board(num)
    count = 0
    success = False
    while not success:
        pos_idx = b.beliefs.index(max(b.beliefs))
        pos = [pos_idx // num, pos_idx % num]
        print("search:", pos_idx)
        success = b.search(pos)
        count += 1
    print("find treasure with " + str(count) + " steps")
