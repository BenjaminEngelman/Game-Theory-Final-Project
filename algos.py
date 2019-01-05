import numpy as np
from helper import boltzmann
from mazes import WIDTH, HEIGHT, updateBeliefState
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
    def __init__(self, hidden_layer_size, nnInputSize):
        self.nn = MLPRegressor(hidden_layer_sizes=(hidden_layer_size, ), activation='logistic')
        self.initNN(nnInputSize)

    def initNN(self, nnInputSize):
        xtrain = np.random.rand(256, nnInputSize)
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


def makeNNInput2(beliefState, x, y, obstacles, goalX, goalY):
    return beliefState.reshape(1, -1)

def makeNNInput3(beliefState, x, y, obstacles, goalX, goalY):
    nnInput = np.concatenate((obstacles, getNNEncodedPosition(x, y)))
    return nnInput.reshape(1, -1)

def makeNNInput4(beliefState, x, y, obstacles, goalX, goalY):
    nnInput = np.concatenate((getNNEncodedPosition(x, y), getNNEncodedPosition(goalX, goalY)))
    return nnInput.reshape(1, -1)

def makeNNInput5(beliefState, x, y, obstacles, goalX, goalY):
    a = getNNEncodedPosition(x, y)
    try:
        b = getNNEncodedPosition(goalX, goalX)
    except IndexError as e:
        print(beliefState)
        print(x, y)
        print(obstacles)
        print(goalX, goalY)
        print(WIDTH, HEIGHT)
        raise e
    
    nnInput = np.concatenate((a, b, obstacles))
    return nnInput.reshape(1, -1)


class Algorithm():

    #################################################
    #     EACH ALGORITHM NEEDS TO IMPLEMENT         #
    #   UPDATE AND GETVALUES AND HAVE A TEMP FIELD  #
    #################################################

    # TEMP = ...

    def updateInternalBeliefState(self, obs, reward, newPos, action):
        """
        Updates the internal state of the algorithm. Only overriden by neural algorithms
        """
        pass

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
        values = self.getValues(pos).tolist()
        seq = sorted(values)
        ranks = np.array([seq.index(p) for p in values])

        return ranks

    def getMostProbableAction(self, pos=None):
        """
        Returns the index of the most probable action (used in majority voting)
        """
        return np.argmax(self.getValues(pos))
        # return np.argmax(self.getBoltzmannProbabilities(pos))


class QLearning(Algorithm):
    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        self.maze = maze
        self.beliefState = None # Overridden in neural version when in experiment 2

        self.pos = maze.start

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getQValues(x, y)
    
    def getQValues(self, x, y):
        return NotImplementedError

    def updateQValues(self, x, y, val):
        return NotImplementedError
    
    
    def update(self, reward, newPos, action, obs):
        self.updateInternalBeliefState(obs, reward, newPos, action)

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
    def __init__(self, maze, params, beliefState=None):
        super().__init__(maze, params)
        self.goal = maze.goal
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        self.makeNNInput = params.MAKE_NN_INPUT
        self.nn = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)
        if beliefState is not None:
            self.beliefState = np.copy(beliefState)
        
    def updateInternalBeliefState(self, obs, reward, newPos, action):
        if self.beliefState is not None:
            updateBeliefState(self.maze, self.beliefState, newPos, obs, action)

    def getQValues(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nn.predict(nnInput)[0]
    
    def updateQValues(self, x, y, val):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nn.train(nnInput, val)


class SARSA(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        self.maze = maze
        self.beliefState = None # Overridden in neural version when in experiment 2
        
        self.pos = maze.start
        
    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getQValues(x, y)
    
    def getQValues(self, x, y):
        return NotImplementedError

    def updateQValues(self, x, y, val):
        return NotImplementedError

    def update(self, reward, newPos, action, obs):
        self.updateInternalBeliefState(obs, reward, newPos, action)

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
    def __init__(self, maze, params, beliefState=None):
        super().__init__(maze, params)
        self.goal = maze.goal
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        self.makeNNInput = params.MAKE_NN_INPUT
        self.nn = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)
        if beliefState is not None:
            self.beliefState = np.copy(beliefState)
        
    def updateInternalBeliefState(self, obs, reward, newPos, action):
        if self.beliefState is not None:
            updateBeliefState(self.maze, self.beliefState, newPos, obs, action)

    def getQValues(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nn.predict(nnInput)[0]
    
    def updateQValues(self, x, y, val):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nn.train(nnInput, val)
    
    


def allActionsExcept(action):
    return [x for x in [0, 1, 2, 3] if x != action]

class ACLA(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        self.maze = maze
        self.beliefState = None # Overridden in neural version when in experiment 2

        self.pos = maze.start

    
    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getPValues(x, y)
    
    def getVValue(self, x, y): return NotImplementedError
    def updateVValue(self, x, y, val): return NotImplementedError
    def getPValues(self, x, y): return NotImplementedError
    def updatePValues(self, x, y, values): return NotImplementedError
    
    def update(self, reward, newPos, action, obs):
        self.updateInternalBeliefState(obs, reward, newPos, action)

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
    def __init__(self, maze, params, beliefState=None):
        super().__init__(maze, params)
        self.goal = maze.goal
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        self.makeNNInput = params.MAKE_NN_INPUT
        if beliefState is not None:
            self.beliefState = np.copy(beliefState)
        
        self.nnV = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)
        self.nnP = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)

    def updateInternalBeliefState(self, obs, reward, newPos, action):
        if self.beliefState is not None:
            updateBeliefState(self.maze, self.beliefState, newPos, obs, action)  
    
    def getVValue(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nnV.predict(nnInput)[0][0]
    
    def updateVValue(self, x, y, value):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nnV.train(nnInput, value)
    
    def getPValues(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nnP.predict(nnInput)[0]
    
    def updatePValues(self, x, y, values):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nnP.train(nnInput, values)


    
class QVLearning(Algorithm):

    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        self.maze = maze
        self.beliefState = None # Overridden in neural version when in experiment 2

        self.pos = maze.start

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getQValues(x, y)
    
    def getVValue(self, x, y): return NotImplementedError
    def updateVValue(self, x, y, val): return NotImplementedError
    def getQValues(self, x, y): return NotImplementedError
    def updateQValues(self, x, y, values): return NotImplementedError
    
    def update(self, reward, newPos, action, obs):
        self.updateInternalBeliefState(obs, reward, newPos, action)

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
    def __init__(self, maze, params, beliefState=None):
        super().__init__(maze, params)
        self.goal = maze.goal
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        self.makeNNInput = params.MAKE_NN_INPUT
        if beliefState is not None:
            self.beliefState = np.copy(beliefState)
        
        self.nnV = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)
        self.nnQ = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)

    def updateInternalBeliefState(self, obs, reward, newPos, action):
        if self.beliefState is not None:
            updateBeliefState(self.maze, self.beliefState, newPos, obs, action)

    def getVValue(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nnV.predict(nnInput)[0][0]
    
    def updateVValue(self, x, y, value):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nnV.train(nnInput, value)
        
    def getQValues(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nnQ.predict(nnInput)[0]

    def updateQValues(self, x, y, values):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nnQ.train(nnInput, values)


class ActorCritic(Algorithm):
    def __init__(self, maze, params):
        self.ALPHA = params.ALPHA
        self.BETA = params.BETA
        self.GAMMA = params.GAMMA
        self.TEMP = params.TEMP
        self.maze = maze
        self.beliefState = None # Overridden in neural version when in experiment 2
        
        self.pos = maze.start

    def getValues(self, pos):
        x, y = pos if pos is not None else self.pos
        return self.getPValues(x, y)

    def getVValue(self, x, y): return NotImplementedError
    def updateVValue(self, x, y, val): return NotImplementedError
    def getPValues(self, x, y): return NotImplementedError
    def updatePValues(self, x, y, values): return NotImplementedError

    def update(self, reward, newPos, action, obs):
        self.updateInternalBeliefState(obs, reward, newPos, action)

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
    def __init__(self, maze, params, beliefState=None):
        super().__init__(maze, params)
        self.goal = maze.goal
        self.obstacles = getNNEncodedObstacles(maze.obstacles)
        self.makeNNInput = params.MAKE_NN_INPUT
        if beliefState is not None:
            self.beliefState = np.copy(beliefState)
        
        self.nnV = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)
        self.nnP = ScikitNeuralNetwork(params.NUM_HIDDEN_NODES, params.NN_INPUT_SIZE)
    
    def updateInternalBeliefState(self, obs, reward, newPos, action):
        if self.beliefState is not None:
            updateBeliefState(self.maze, self.beliefState, newPos, obs, action)

    def getVValue(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nnV.predict(nnInput)[0][0]
    
    def updateVValue(self, x, y, value):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nnV.train(nnInput, value)
    
    def getPValues(self, x, y):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        return self.nnP.predict(nnInput)[0]
    
    def updatePValues(self, x, y, values):
        goalX, goalY = self.goal
        nnInput = self.makeNNInput(self.beliefState, x, y, self.obstacles, goalX, goalY)
        self.nnP.train(nnInput, values)
    
    



























