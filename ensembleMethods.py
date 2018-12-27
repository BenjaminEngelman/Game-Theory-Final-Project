import numpy as np
from collections import Counter
from helper import boltzmann


def boltzmannChoice(values, temp):
    probabilities = boltzmann(values, temp)
    return np.random.choice([0, 1, 2, 3], p=probabilities)
    
def simpleChoice(values, temp):
    transformedValues = np.power(values, 1 / temp)
    probabilities = transformedValues / transformedValues.sum()
    return np.random.choice([0, 1, 2, 3], p=probabilities)


def majorityVote(algorithms, temp):
    bestActions = [algo.getMostProbableAction() for algo in algorithms]
    counter = Counter(bestActions)
    counts = np.array([counter[action] for action in [0, 1, 2, 3]])
    return boltzmannChoice(counts, temp)
    

def rankVote(algorithms, temp):
    allRanks = np.array([algo.getActionRanking() for algo in algorithms])
    probabilities = allRanks.sum(axis = 0)
    return boltzmannChoice(probabilities, temp)



def boltzmannMultVote(algorithms, temp):
    actionsProbabilities = np.array([algo.getBoltzmannProbabilities() for algo in algorithms])
    prefs = np.prod(actionsProbabilities, axis=0)
    return simpleChoice(prefs, temp)

def boltzmannAddVote(algorithms, temp):
    actionsProbabilities = np.array([algo.getBoltzmannProbabilities() for algo in algorithms])
    prefs = np.sum(actionsProbabilities, axis=0)
    return simpleChoice(prefs, temp)



