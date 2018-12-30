import numpy as np
from helper import boltzmann
from mazes import WIDTH, HEIGHT




class Algorithm():

    #################################################
    #     EACH ALGORITHM NEEDS TO IMPLEMENT         #
    #   UPDATE AND GETVALUES AND HAVE A TEMP FIELD  #
    #################################################

    # TEMP = ...

    def update(self, reward, newPos, action):
        """
        Updates the internal state of the algorithm.
        """
        return NotImplementedError

    def getValues(self, pos):
        """
        Returns the value of each action, which is different for each algorithm,
        and represents how good each action is.
        """
        return NotImplementedError

    def getActionBoltzmannChoice(self):
        """
        Choose an action using the Boltzmann probabilities
        """
        return np.random.choice([0,1,2,3], p=self.getBoltzmannProbabilities())

    def getBoltzmannProbabilities(self, pos=None):
        """
        Returns a list containing each action's probability 
        (which is calculated using Boltzmann).
        Used in Boltzmann addition and Boltzmann multiplication.
        """
        values = self.getValues(pos)
        return boltzmann(values, self.TEMP)

    def getActionRanking(self, pos=None):
        """
        Returns a list containing the ranking of each action (used in rank voting).
        """
        probabilities = self.getBoltzmannProbabilities(pos)
        seq = sorted(probabilities)
        ranks = np.array([seq.index(p) for p in probabilities])
        return ranks

    def getMostProbableAction(self, pos=None):
        """
        Returns the index of the most probable action (used in majority voting)
        """
        return np.argmax(self.getBoltzmannProbabilities(pos))


class QLearning(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP

        self.pos = maze.start
        self.qValues = np.zeros(shape=(WIDTH, HEIGHT, 4))

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.qValues[x, y]

    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos
        bestQValueInNextPos = np.max(self.qValues[newX, newY])
        self.qValues[oldX, oldY, action] += self.ALPHA * \
            (reward + self.GAMMA * bestQValueInNextPos -
             self.qValues[oldX, oldY, action])
        self.pos = newPos

        # Pseudo-code for neural network version
        # bestQValueInNextPos = np.max(self.nnQValues(newX, newY))
        # targetQValues = self.nnQValues(oldX, oldY)
        # targetQValues[action] = self.ALPHA * (reward + self.GAMMA * bestQValueInNextPos - targetQValues[action])
        # nn.train(oldX, oldY, targetQValues)


class SARSA(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP

        self.pos = maze.start
        self.qValues = np.zeros(shape=(WIDTH, HEIGHT, 4))

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.qValues[x, y]

    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos

        # To avoid recusion, we compute the next action using only the SARSA algorithm and
        # not the whole ensemble
        nextAction = self.getMostProbableAction(newPos)

        qValueOfNextAction = self.qValues[newX, newY, nextAction]
        self.qValues[oldX, oldY, action] += self.ALPHA * \
            (reward + self.GAMMA * qValueOfNextAction -
             self.qValues[oldX, oldY, action])

        self.pos = newPos


def allActionsExcept(action):
    return [x for x in [0, 1, 2, 3] if x != action]

class ACLA(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP

        self.pos = maze.start
        self.vValues = np.zeros(shape=(WIDTH, HEIGHT))
        self.pValues = np.zeros(shape=(WIDTH, HEIGHT, 4))
    
    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.pValues[x, y]
    
    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos

        # Update the v values
        self.vValues[oldX, oldY] += self.BETA * (reward + self.GAMMA * self.vValues[newX, newY] - self.vValues[oldX, oldY])
        

        # Update the P values
        delta = self.GAMMA * self.vValues[newX, newY] + reward - self.vValues[oldX, oldY]  
        if delta >= 0:
            self.pValues[oldX, oldY, action] += self.ALPHA * (1 - self.pValues[oldX, oldY, action])
            for a in allActionsExcept(action):
                self.pValues[oldX, oldY, a] += self.ALPHA * (0 - self.pValues[oldX, oldY, a])

        else:

            self.pValues[oldX, oldY, action] -= self.ALPHA * self.pValues[oldX, oldY, action]
            
            # Add ALPHA * fraction for all actions except the action that was used
            pValuesSum = self.pValues[oldX, oldY].sum()
            denom = pValuesSum - self.pValues[oldX, oldY, action]

            for a in allActionsExcept(action):
                if denom > 0:
                    # Normal case:
                    num = self.pValues[oldX, oldY, a]
                    self.pValues[oldX, oldY, a] += self.ALPHA * ((num / denom) - self.pValues[oldX, oldY, a]) 
                else:
                    # Special Rule 1: if denom is <= 0, put 1/3
                    self.pValues[oldX, oldY, a] = 1 / (4 - 1)
                    
        # Special rule 2: p values must be between 0 and 1
        self.pValues[oldX, oldY, :] =  np.clip(self.pValues[oldX, oldY], 0, 1)
            
        self.pos = newPos


    
class QVLearning(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP

        self.pos = maze.start
        self.qValues = np.zeros(shape=(WIDTH, HEIGHT, 4))
        self.vValues = np.zeros(shape=(WIDTH, HEIGHT))

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.qValues[x, y]

    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos

        self.vValues[oldX, oldY] += self.BETA * \
            (reward + self.GAMMA *
             self.vValues[newX, newY] - self.vValues[oldX, oldY])
        self.qValues[oldX, oldY, action] += self.ALPHA * \
            (reward + self.GAMMA *
             self.vValues[newX, newY] - self.qValues[oldX, oldY, action])

        self.pos = newPos


class ActorCritic(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        
        self.pos = maze.start
        self.vValues = np.zeros(shape=(WIDTH, HEIGHT))
        self.pValues = np.zeros(shape=(WIDTH, HEIGHT, 4))

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.pValues[x, y]

    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos

        self.vValues[oldX, oldY] += self.BETA * \
            (reward + self.GAMMA *
             self.vValues[newX, newY] - self.vValues[oldX, oldY])
        self.pValues[oldX, oldY, action] += self.ALPHA * \
            (reward + self.GAMMA *
             self.vValues[newX, newY] - self.vValues[oldX, oldY])

        self.pos = newPos





























