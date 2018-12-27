from algos import *
from ensembleMethods import *

class Agent():

    def __init__(self, maze, ensembleMethod):
        self.maze = maze
        self.ensembleMethod = ensembleMethod
        self.Algos = [QLearning(maze), SARSA(maze), ActorCritic(maze), QVLearning(maze)]
    
    def run(self):
        pass