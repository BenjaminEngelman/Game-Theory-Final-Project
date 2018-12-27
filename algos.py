import numpy as np

def boltzmann(values, temp):
    boltzmannValues = np.exp(values / temp)
    boltzmannProbabilities = boltzmannValues / boltzmannValues.sum()
    return boltzmannProbabilities


class MazeProperties:
    HEIGHT = 10
    WIDTH = 5
    STARTPOS = (2, 3)

class Algorithm():

    ##############################################
    #   EACH ALGORITHM NEEDS TO IMPLEMENT        #
    #   THESE TWO METHODS AND HAVE A TEMP FIELD  #
    ##############################################

    # TEMP = ... 

    def update(self, reward, newPos, action):
        """
        Updates the internal state of the algorithm.
        """
        return NotImplementedError
    
    def getValues(self):
        """
        Returns the value of each action, which is different for each algorithm,
        and represents how good each action is.
        """
        return NotImplementedError










    def getBoltzmannProbabilities(self, pos):
        """
        Returns a list containing each action's probability 
        (which is calculated using Boltzmann).
        Used in Boltzmann addition and Boltzmann multiplication.
        """
        values = self.getValues(pos)
        return boltzmann(values, self.TEMP)
    
    def getActionRanking(self, pos):
        """
        Returns a list containing the ranking of each action (used in rank voting).
        """
        probabilities = self.getBoltzmannProbabilities(pos)
        seq = sorted(probabilities)
        ranks = [seq.index(p) for p in probabilities]
        return ranks
        

    def getMostProbableAction(self, pos):
        """
        Returns the index of the most probable action (used in majority voting)
        """
        return np.argmax(self.getBoltzmannProbabilities(pos))








class QLearning(Algorithm):
    ALPHA = 0.2
    GAMMA = 0.9
    TEMP = 1

    def __init__(self, maze):
        self.pos = maze.start
        self.qValues = np.zeros(shape=(maze.WIDTH, maze.HEIGHT, 4))
    
    def getValues(self, pos):
        x, y = pos
        return self.qValues[x, y]
    
    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos
        bestQValueInNextPos = np.max(self.qValues[newX, newY])
        self.qValues[oldX, oldY, action] += self.ALPHA * (reward + self.GAMMA * bestQValueInNextPos - self.qValues[oldX, oldY, action]) 
        self.pos = newPos

        # Pseudo-code for neural network version
        # bestQValueInNextPos = np.max(self.nnQValues(newX, newY))
        # targetQValues = self.nnQValues(oldX, oldY)
        # targetQValues[action] = self.ALPHA * (reward + self.GAMMA * bestQValueInNextPos - targetQValues[action]) 
        # nn.train(oldX, oldY, targetQValues)
        







class SARSA(Algorithm):
    ALPHA = 0.2
    GAMMA = 0.9
    TEMP = 1

    def __init__(self, maze):
        self.pos = maze.start
        self.qValues = np.zeros(shape=(maze.WIDTH, maze.HEIGHT, 4))
    
    def getValues(self, pos):
        x, y = pos
        return self.qValues[x, y]
    
    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos
        
        # To avoid recusion, we compute the next action using only the SARSA algorithm and
        # not the whole ensemble
        nextAction = self.getMostProbableAction(newPos)
        
        qValueOfNextAction = self.qValues[newX, newY, nextAction]
        self.qValues[oldX, oldY, action] += self.ALPHA * (reward + self.GAMMA * qValueOfNextAction - self.qValues[oldX, oldY, action]) 
        
        self.pos = newPos



    































