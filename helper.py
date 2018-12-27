import numpy as np

def boltzmann(values, temp):
    boltzmannValues = np.exp(values / temp)
    boltzmannProbabilities = boltzmannValues / boltzmannValues.sum()
    return boltzmannProbabilities