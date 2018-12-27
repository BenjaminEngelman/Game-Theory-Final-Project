import numpy as np
from collections import Counter
from helper import boltzmann


def boltzmannChoice(values, temp):
    probabilities = boltzmann(values, temp)
    return np.random.choice([0, 1, 2, 3], p=probabilities)
    
def simpleChoice(values):
    pass


def majorityVote(algorithms):
    bestActions = [algo.getMostProbableAction() for algo in algorithms]
    counter = Counter(bestActions)
    counts = [counter[action] for action in [0, 1, 2, 3]]
    return boltzmannChoice(counts)






def boltzmannMultVoting(algorithms):
    actionsProbabilities = np.array([algo.getBoltzmannProbabilities() for algo in algorithms])
    prefs = np.prod(actionsProbabilities, axis=0)



