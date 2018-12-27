import numpy as np
from collections import Counter
from helper import boltzmann


def boltzmannChoice(values):
    probabilities = boltzmann(values)
    return np.random.choice([0, 1, 2, 3], p=probabilities)
    

def majorityVote(algorithms):
    bestActions = [algo.getMostProbableAction() for algo in algorithms]
    counter = Counter(bestActions)
    counts = [counter[action] for action in [0, 1, 2, 3]]
    return boltzmannChoice(counts)


