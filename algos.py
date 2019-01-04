import numpy as np
from helper import boltzmann
from mazes import WIDTH, HEIGHT
from sklearn.neural_network import MLPRegressor



       
def getNNEncodedObstacles(obstacles):
    nnObstacles = np.zeros(shape=(WIDTH, HEIGHT))
    for (x, y) in obstacles:
        nnObstacles[x, y] = 1
    return nnObstacles.reshape(-1)

def getNNEncodedPosition(x, y):
    nnPosition = np.zeros(shape=(WIDTH, HEIGHT))
    nnPosition[x, y] = 1
    return nnPosition.reshape(-1)

class ScikitNeuralNetwork():
    def __init__(self, hidden_layer_size=60):
        self.nn = MLPRegressor(hidden_layer_sizes=(hidden_layer_size, ), activation='logistic')
        self.initNN()

    def initNN(self):
        xtrain = np.random.rand(256, 2 * WIDTH * HEIGHT)
        ytrain = np.random.rand(256, 4)
        self.nn.fit(xtrain, ytrain)

    def predict(self, x):
        return self.nn.predict(x)
    
    def train(self, x, val):
        self.nn.partial_fit(x, [val])

class KerasNeuralNetwork():
    def __init__(self):
        from keras.models import Sequential
        from keras.layers import Dense
        import keras
        self.nn = Sequential()
        self.nn.add(Dense(units=60, activation='sigmoid', input_dim=2 * WIDTH * HEIGHT))
        self.nn.add(Dense(units=4, activation='sigmoid'))
        self.nn.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
            metrics=['accuracy']
        )
        print("Model compiled and ready to run!")

    def predict(self, x):
        return self.nn.predict(x) #[0]
    
    def train(self, x, val):
        self.nn.train_on_batch(x, np.array([val]))


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
        probabilities = self.getBoltzmannProbabilities(pos).tolist()
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

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getQValues(x, y)
    
    def getQValues(self, x, y):
        return NotImplementedError

    def updateQValues(self, x, y, val):
        return NotImplementedError
    
    
    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos
        bestQValueInNextPos = np.max(self.getQValues(newX, newY))
        QValues = self.getQValues(oldX, oldY)
        QValues[action] += self.ALPHA * (reward + self.GAMMA * bestQValueInNextPos - QValues[action])
        self.updateQValues(oldX, oldY, QValues)
        self.pos = newPos

class QLearningNormal(QLearning):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.qValues = np.zeros(shape=(WIDTH, HEIGHT, 4))

    def getQValues(self, x, y):
        return self.qValues[x, y]

    def updateQValues(self, x, y, val):
        self.qValues[x, y] = val


class QLearningNeuronal(QLearning):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.nn = ScikitNeuralNetwork()
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        

    def getNNInput(self, x, y):
        nnInput = np.concatenate((self.obstacles, getNNEncodedPosition(x, y)))
        return nnInput.reshape(1, -1)
        
    def getQValues(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nn.predict(nnInput)[0]
    
    def updateQValues(self, x, y, val):
        nnInput = self.getNNInput(x, y)
        self.nn.train(nnInput, val)


class SARSA(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        
        self.pos = maze.start
        
    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getQValues(x, y)
    
    def getQValues(self, x, y):
        return NotImplementedError

    def updateQValues(self, x, y, val):
        return NotImplementedError

    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos

        # To avoid recusion, we compute the next action using only the SARSA algorithm and
        # not the whole ensemble
        nextAction = self.getMostProbableAction(newPos)

        qValueOfNextAction = self.getQValues(newX, newY)[nextAction]
        Qvalues = self.getQValues(oldX, oldY)
        Qvalues[action] += self.ALPHA * \
            (reward + self.GAMMA * qValueOfNextAction -
             Qvalues[action])
        self.updateQValues(oldX, oldY, Qvalues)
        self.pos = newPos

class SARSANormal(SARSA):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.qValues = np.zeros(shape=(WIDTH, HEIGHT, 4))

    def getQValues(self, x, y):
        return self.qValues[x, y]

    def updateQValues(self, x, y, val):
        self.qValues[x, y] = val


class SARSANeuronal(SARSA):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.nn = ScikitNeuralNetwork()
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        

    def getNNInput(self, x, y):
        nnInput = np.concatenate((self.obstacles, getNNEncodedPosition(x, y)))
        return nnInput.reshape(1, -1)
        
    def getQValues(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nn.predict(nnInput)[0]
    
    def updateQValues(self, x, y, val):
        nnInput = self.getNNInput(x, y)
        self.nn.train(nnInput, val)
    
    


def allActionsExcept(action):
    return [x for x in [0, 1, 2, 3] if x != action]

class ACLA(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP

        self.pos = maze.start

    
    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getPValues(x, y)
    
    def getVValue(self, x, y): return NotImplementedError
    def updateVValue(self, x, y, val): return NotImplementedError
    def getPValues(self, x, y): return NotImplementedError
    def updatePValues(self, x, y, values): return NotImplementedError
    
    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos
        
        oldVValue = self.getVValue(oldX, oldY)
        newVValue = self.getVValue(newX, newY)

        # Update the v values
        oldVValue += self.BETA * (reward + self.GAMMA * newVValue - oldVValue)
        self.updateVValue(oldX, oldY, oldVValue)

        # Update the P values
        delta = self.GAMMA * newVValue + reward - oldVValue 
        oldPValues = self.getPValues(oldX, oldY)
        if delta >= 0:
            oldPValues[action] += self.ALPHA * (1 - oldPValues[action])
            for a in allActionsExcept(action):
                oldPValues[a] += self.ALPHA * (0 - oldPValues[a])

        else:

            oldPValues[action] -= self.ALPHA * oldPValues[action]
            
            # Add ALPHA * fraction for all actions except the action that was used
            pValuesSum = oldPValues.sum()
            denom = pValuesSum - oldPValues[action]

            for a in allActionsExcept(action):
                if denom > 0:
                    # Normal case:
                    num = oldPValues[a]
                    oldPValues[a] += self.ALPHA * ((num / denom) - oldPValues[a]) 
                else:
                    # Special Rule 1: if denom is <= 0, put 1/3
                    oldPValues[a] = 1 / (4 - 1)
                    
        # Special rule 2: p values must be between 0 and 1
        oldPValues = np.clip(oldPValues, 0, 1)
        self.updatePValues(oldX, oldY, oldPValues)
            
        self.pos = newPos

class ACLANormal(ACLA):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.vValues = np.zeros(shape=(WIDTH, HEIGHT))
        self.pValues = np.zeros(shape=(WIDTH, HEIGHT, 4))
        
        
    def getVValue(self, x, y):
        return self.vValues[x, y]
    
    def updateVValue(self, x, y, value):
        self.vValues[x, y] = value
    
    def getPValues(self, x, y):
        return self.pValues[x, y]
    
    def updatePValues(self, x, y, values):
        self.pValues[x, y] = values

    
class ACLANeuronal(ACLA):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.nnV = ScikitNeuralNetwork()
        self.nnP = ScikitNeuralNetwork()
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
    
    def getNNInput(self, x, y):
        nnInput = np.concatenate((self.obstacles, getNNEncodedPosition(x, y)))
        return nnInput.reshape(1, -1)
    
    def getVValue(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nnV.predict(nnInput)[0][0]
    
    def updateVValue(self, x, y, value):
        nnInput = self.getNNInput(x, y)
        self.nnV.train(nnInput, value)
    
    def getPValues(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nnP.predict(nnInput)[0]
    
    def updatePValues(self, x, y, values):
        nnInput = self.getNNInput(x, y)
        self.nnP.train(nnInput, values)


    
class QVLearning(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP

        self.pos = maze.start

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getQValues(x, y)
    
    def getVValue(self, x, y): return NotImplementedError
    def updateVValue(self, x, y, val): return NotImplementedError
    def getQValues(self, x, y): return NotImplementedError
    def updateQValues(self, x, y, values): return NotImplementedError
    
    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos
        
        VValue = self.getVValue(oldX, oldY)
        VValue += self.BETA * (reward + self.GAMMA * self.getVValue(newX, newY) - self.getVValue(oldX, oldY))
        self.updateVValue(oldX, oldY, VValue)
        
        QValues = self.getQValues(oldX, oldY)
        QValues[action] += self.ALPHA * (reward + self.GAMMA * self.getVValue(newX, newY) - self.getQValues(oldX, oldY)[action])
        self.updateQValues(oldX, oldY, QValues)
        self.pos = newPos

class QVLearningNormal(QVLearning):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.qValues = np.zeros(shape=(WIDTH, HEIGHT, 4))
        self.vValues = np.zeros(shape=(WIDTH, HEIGHT))
    
    def getVValue(self, x, y):
        return self.vValues[x, y]

    def updateVValue(self, x, y, val):
        self.vValues[x, y] = val

    def getQValues(self, x, y):
        return self.qValues[x, y]

    def updateQValues(self, x, y, val):
        self.qValues[x, y] = val
    



class QVLearningNeuronal(QVLearning):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.nnV = ScikitNeuralNetwork()
        self.nnQ = ScikitNeuralNetwork()

        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        

    def getNNInput(self, x, y):
        nnInput = np.concatenate((self.obstacles, getNNEncodedPosition(x, y)))
        return nnInput.reshape(1, -1)
    
    def getVValue(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nnV.predict(nnInput)[0][0]
    
    def updateVValue(self, x, y, value):
        nnInput = self.getNNInput(x, y)
        self.nnV.train(nnInput, value)
        
    def getQValues(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nnQ.predict(nnInput)[0]

    def updateQValues(self, x, y, values):
        nnInput = self.getNNInput(x, y)
        self.nnQ.train(nnInput, values)


class ActorCritic(Algorithm):
    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        
        self.pos = maze.start

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getPValues(x, y)

    def getVValue(self, x, y): return NotImplementedError
    def updateVValue(self, x, y, val): return NotImplementedError
    def getPValues(self, x, y): return NotImplementedError
    def updatePValues(self, x, y, values): return NotImplementedError

    def update(self, reward, newPos, action):
        oldX, oldY = self.pos
        newX, newY = newPos

        oldVValue = self.getVValue(oldX, oldY)
        newPosVValue = self.getVValue(newX, newY)
        updatedVValue = oldVValue + self.BETA * (reward + self.GAMMA * newPosVValue - oldVValue)
        self.updateVValue(oldX, oldY, updatedVValue)

        pValues = self.getPValues(oldX, oldY)
        pValues[action] += self.ALPHA * (reward + self.GAMMA * newPosVValue - oldVValue)
        self.updatePValues(oldX, oldY, pValues)

        self.pos = newPos

class ActorCriticNormal(ActorCritic):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.vValues = np.zeros(shape=(WIDTH, HEIGHT))
        self.pValues = np.zeros(shape=(WIDTH, HEIGHT, 4))
        
        
    def getVValue(self, x, y):
        return self.vValues[x, y]
    
    def updateVValue(self, x, y, value):
        self.vValues[x, y] = value
    
    def getPValues(self, x, y):
        return self.pValues[x, y]
    
    def updatePValues(self, x, y, values):
        self.pValues[x, y] = values

    

class ActorCriticNeuronal(ActorCritic):
    def __init__(self, maze, params):
        super().__init__(maze, params)
        self.nnV = ScikitNeuralNetwork()
        self.nnP = ScikitNeuralNetwork()
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
    
    def getNNInput(self, x, y):
        nnInput = np.concatenate((self.obstacles, getNNEncodedPosition(x, y)))
        return nnInput.reshape(1, -1)
    
    def getVValue(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nnV.predict(nnInput)[0][0]
    
    def updateVValue(self, x, y, value):
        nnInput = self.getNNInput(x, y)
        self.nnV.train(nnInput, value)
    
    def getPValues(self, x, y):
        nnInput = self.getNNInput(x, y)
        return self.nnP.predict(nnInput)[0]
    
    def updatePValues(self, x, y, values):
        nnInput = self.getNNInput(x, y)
        self.nnP.train(nnInput, values)
    
    



























