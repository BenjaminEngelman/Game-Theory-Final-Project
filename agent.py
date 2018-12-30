from algos import *
from ensembleMethods import *
from mazes import *
import time

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

        for i in range(episodes):
            episodeReward = 0
            while not self.maze.isDone():
                action = self.ensembleMethod(self.algos, self.temp)
                reward, new_state = self.maze.step(action)
                episodeReward += reward
                # maze.render()
                                
                for algo in self.algos:
                    algo.update(reward, new_state, action)
                
            print("Done %d in %d steps" % (i + 1, self.maze.actions_counter))
            rewardsOverTime.append(episodeReward)
            maze.reset()

        return rewardsOverTime


if __name__ == "__main__":
    parmasList = [
        AlgoParams(alpha=0.2, gamma=0.9, temp=1),
        AlgoParams(alpha=0.2, gamma=0.9, temp=1),
        AlgoParams(alpha=0.1, beta=0.2, gamma=0.95, temp=1),
        AlgoParams(alpha=0.2, beta=0.2, gamma=0.9, temp=1),
        AlgoParams(alpha=0.005, beta=0.1, gamma=0.99, temp=1/9)
    ]

    maze = createSimpleMaze()
    agent = Agent(maze, majorityVote, parmasList, temp=1/1.6)

    start = time.time()
    rewardsOverTime = agent.learn(5000)
    print("Learning process took %d seconds" % (time.time() - start))

    print(rewardsOverTime[-10:])

    
    # reward intake = reward moyen par mouvement
    # Il mesure deux choses
    # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
