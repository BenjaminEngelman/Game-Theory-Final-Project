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

    def update(self, reward, new_state, action):
        raise NotImplementedError

    def chooseAction(self):
        raise NotImplementedError

    def learn(self, episodes):
        last2500RewardIntakes = np.zeros(2500)
        rewardIntakesEvery2500Episodes = np.zeros(episodes // 2500)
        allRewardIntakes = np.zeros(50000)
        numberOfSteps = np.zeros(50000)

        for episodeNum in range(episodes):
            episodeReward = 0
            while not self.maze.isDone():
                action = self.chooseAction()
                reward, new_state = self.maze.step(action)
                episodeReward += reward
                # maze.render()
                self.update(reward, new_state, action)

            allRewardIntakes[episodeNum] = (episodeReward / maze.actions_counter)
            numberOfSteps[episodeNum] = maze.actions_counter

            if episodeNum % 2500 == 2499:
                print("Done %d in %d steps" %
                      (episodeNum + 1, self.maze.actions_counter))
                
                rewardIntakesEvery2500Episodes[episodeNum // 2500] = (episodeReward / maze.actions_counter)

            if episodeNum + 2500 >= episodes:
                last2500RewardIntakes[episodeNum - 47500] = (episodeReward / maze.actions_counter)

            maze.reset()

        return allRewardIntakes, numberOfSteps, last2500RewardIntakes.mean(), rewardIntakesEvery2500Episodes.sum()


class AgentWithSingleAlgo(Agent):
    def __init__(self, maze, algo, params):
        self.maze = maze
        self.algo = algo(maze, params)
    
    def update(self, reward, new_state, action):
        self.algo.update(reward, new_state, action)

    def chooseAction(self):
        return self.algo.getMostProbableAction()


class AgentWithEnsemble(Agent):

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

    def update(self, reward, new_state, action):
        for algo in self.algos:
            algo.update(reward, new_state, action)

    def chooseAction(self):
        return self.ensembleMethod(self.algos, self.temp)


if __name__ == "__main__":
    paramsDict = {
        "Q-Learning": AlgoParams(alpha=0.2, gamma=0.9, temp=1),
        "SARSA": AlgoParams(alpha=0.2, gamma=0.9, temp=1),
        "Actor-Critic": AlgoParams(alpha=0.1, beta=0.2, gamma=0.95, temp=1),
        "QV-Learning": AlgoParams(alpha=0.2, beta=0.2, gamma=0.9, temp=1),
        "ACLA": AlgoParams(alpha=0.005, beta=0.1, gamma=0.99, temp=1/9)
    }

    maze = createSimpleMaze()
    agent = AgentWithSingleAlgo(
        maze, QLearning, paramsDict["Q-Learning"])

    start = time.time()

    # reward intake = reward moyen par mouvement
    # Il mesure deux choses
    # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
    allRewardIntakes, numberOfSteps, final, cumulative = agent.learn(50000)
    print("Learning process took %d seconds" % (time.time() - start))

    print("Final: %s, Cumulative: %s" % (final, cumulative))
    saveToFile("data_QLearning.json", [final, cumulative])
    saveToFile("allRewardsIntakes_QLearning.json", allRewardIntakes)
    saveToFile("numberOfSteps_QLearning.json", numberOfSteps)
