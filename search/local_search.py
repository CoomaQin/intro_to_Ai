from search.search_algs import DFS
import numpy as np
from search.matrix import generate_maze
from tqdm import tqdm
import copy


# Genetic Algorithm

class maze:
    def __init__(self, matrix, hardness):
        self.matrix = matrix
        self.hardness = hardness


def genetic(population_size, p, dim, terminate, search, mutation_rate):
    '''
    :param mutation_rate: mutation rate in (0, 1)
    :param search: the global search alg for finding maze solutions
    :param population_size: the size of population
    :param p: the probability of cells being blocked, in (0, 1)
    :param dim: the size of each maze
    :param terminate: terminal condition, times that the population would evolve
    :return: a tuple of maze objects
    '''
    population = []
    # the initial population
    for _ in range(population_size):
        m = generate_maze(p, dim)
        tmp = copy.copy(m)
        h = compute_hardness(tmp, search)
        mz = maze(m, h)
        population.append(mz)
    for i in range(terminate):
        population = sorted(population, key=lambda x: x.hardness, reverse=True)
        next_population = []
        # extend a part of the current population to the next population
        stay_num = int(0.1 * population_size)
        next_population.extend(population[:stay_num])
        #select 15 most fit children and breed them
        for _ in tqdm(range(population_size - stay_num), desc=str(i + 1) + ' generation'):
            rv = np.random.random_integers(15, size=2)
            par1 = population[rv[0]]
            par2 = population[rv[1]]
            child = gene_crossover(par1, par2, dim, search, mutation_rate, [rv[0], rv[1]])
            # if i == 1:
            #     print(child.matrix)
            next_population.append(child)
        population = next_population
        # if we get to a satisfying hardness end early
        if population[0].hardness >= 6800:
            break
    return sorted(population, key=lambda x: x.hardness, reverse=True)

#return how many nodes a given alg visits before reaching the end
def compute_hardness(m, search):
    tmp = copy.copy(m)
    tmp, _ = search(m)
    return len(tmp[tmp == 1])

#randomly mutate ganes at a given rate
def gene_mute(m, rate):
    m = np.array(m, dtype=int)
    for i in range(len(m)):
        for j in range(len(m)):
            r = np.random.random_sample()
            if m[i, j] == 0 and (i != 0 and j != 0):
                if r < rate:
                    m[i, j] = 2
            else:
                if m[i, j] == 2:
                    if r < rate:
                        m[i, j] = 0
    return m

# we partition the matrices and randomly mix them and then mutate a random part of the matrix
#the descebdabts top 5 specimen do not mutate
def gene_crossover(par1, par2, dim, search, mutation, positions):
    m1 = par1.matrix
    m2 = par2.matrix
    child_matrix = np.zeros([dim, dim])
    # make sure the child is solvable
    success = False
    # the dim of each partition
    size = 10
    # Combine features of parents and mutation based on partitioned matrices
    partition = int(dim / size)
    while not success:
        for i in range(partition):
            for j in range(partition):
                r = np.random.random_sample()
                #rmute = np.random.random_sample()
                flip = np.random.random_sample()
                if r < .5:
                    child_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size] = m1[i * size:(i + 1) * size,
                                                                                     j * size:(j + 1) * size]
                else:
                    child_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size] = m2[i * size:(i + 1) * size,
                                                                                     j * size:(j + 1) * size]
                # only mutate mazes that aren't the x most fit ones
                if positions[0] >= 4 or positions[1] >= 4:
                    if flip < .5:
                        child_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size] = gene_mute(
                            child_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size], mutation)
                    else:
                        child_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size] = gene_mute(
                            child_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size], mutation)
        tmp_mz = copy.copy(child_matrix)
        _, success = search(tmp_mz)
    h = len(tmp_mz[tmp_mz == 1])
    return maze(child_matrix, h)