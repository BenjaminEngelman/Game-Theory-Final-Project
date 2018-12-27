import numpy as np


def rankVote(algorithms, temp):
    allRanks = np.array([algo.getActionRanking() for algo in algorithms])
    probailities = allRanks.sum(axis = 0)
    return boltzmannChoice(probailities, temp)


    