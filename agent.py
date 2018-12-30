from algos import *
from ensembleMethods import *
from mazes import *
from helper import *
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
        last2500RewardIntakes = np.zeros(2500)
        rewardIntakesEvery2500Episodes = np.zeros(episodes // 2500)
        

        for episodeNum in range(episodes):
            episodeReward = 0
            while not self.maze.isDone():
                action = self.ensembleMethod(self.algos, self.temp)
                reward, new_state = self.maze.step(action)
                episodeReward += reward
                # maze.render()
                                
                for algo in self.algos:
                    algo.update(reward, new_state, action)
                
            if episodeNum % 2500 == 2499:
                print("Done %d in %d steps" % (episodeNum + 1, self.maze.actions_counter))
                rewardIntakesEvery2500Episodes[episodeNum // 2500] = (episodeReward / maze.actions_counter)

            if episodeNum + 2500 >= episodes:
                last2500RewardIntakes[episodeNum - 47500].append(episodeReward / maze.actions_counter)
            
            maze.reset()

        return last2500RewardIntakes.mean(), rewardIntakesEvery2500Episodes.sum()


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

    # reward intake = reward moyen par mouvement
    # Il mesure deux choses
    # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
    final, cumulative = agent.learn(50000)
    print("Learning process took %d seconds" % (time.time() - start))

    print("Final: %s, Cumulative: %s" % (final, cumulative))
    saveToFile("data.json", [final, cumulative])

    
    
