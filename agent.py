from algos import *
from ensembleMethods import *
from mazes import SimpleMaze


class AlgoParams():
    def __init__(self, alpha=None, beta=None, gamma=None, temp=None):
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.TEMP = temp


class Agent():

    def __init__(self, maze, ensembleMethod, algoParamsList, temp):
        self.maze = maze
        self.temp = temp
        self.ensembleMethod = ensembleMethod
        self.algos = [
            QLearning(maze, algoParamsList[0]),
            SARSA(maze, algoParamsList[1]),
            ActorCritic(maze, algoParamsList[2]),
            QVLearning(maze, algoParamsList[3]),
            ACLA(maze, algoParamsList[4])
        ]

    def learn(self, episodes):
        rewardsOverTime = []

        for _ in range(episodes):
            episodeReward = 0
            while not self.maze.isDone():
                action = self.ensembleMethod(self.algos, self.temp)
                reward, new_state = self.maze.step(action)
                episdodeReward += reward

                for algo in self.algos:
                    algo.update(reward, new_state, action)

            rewardsOverTime.append(episodeReward)

        return rewardsOverTime


if __name__ == "__main__":
    parmasList = [
        AlgoParams(alpha=0.2, gamma=0.9, temp=1),
        AlgoParams(alpha=0.2, gamma=0.9, temp=1),
        AlgoParams(alpha=0.1, beta=0.2, gamma=0.95, temp=1),
        AlgoParams(alpha=0.2, beta=0.2, gamma=0.9, temp=1),
        AlgoParams(alpha=0.005, beta=0.1, gamma=0.99, temp=1/9)
    ]

    maze = SimpleMaze()
    agent = Agent(maze, majorityVote, parmasList, temp=1/1.6)
    rewardsOverTime = agent.learn(50000)
