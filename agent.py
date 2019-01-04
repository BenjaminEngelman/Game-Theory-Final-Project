from algos import *
from ensembleMethods import *
from mazes import *
from helper import *
import time


class AlgoParams():
    def __init__(self, alpha=None, beta=None, gamma=None, temp=None, numHiddenNodes=None):
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.TEMP = temp
        self.NUM_HIDDEN_NODES = numHiddenNodes


class Agent():

    def update(self, reward, new_state, action):
        raise NotImplementedError

    def chooseAction(self):
        raise NotImplementedError

    def learn(self, episodes):
        last2500RewardIntakes = np.zeros(2500)
        rewardIntakesEvery2500Episodes = np.zeros(episodes // 2500)
        allRewardIntakes = np.zeros(episodes)
        numberOfSteps = np.zeros(episodes)

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
            if episodeNum % 100 == 0 : 
                print(episodeNum, self.maze.actions_counter)

            # if episodeNum % 2500 == 2499:
            #     print("Done %d " % (episodeNum + 1))
            #     rewardIntakesEvery2500Episodes[episodeNum // 2500] = (episodeReward / self.maze.actions_counter)

            # if episodeNum + 2500 >= episodes:
            #     last2500RewardIntakes[episodeNum - 47500] = (episodeReward / self.maze.actions_counter)

            self.maze.reset()

        #return last2500RewardIntakes.mean(), rewardIntakesEvery2500Episodes.sum()
        #return allRewardIntakes, numberOfSteps, last2500RewardIntakes.mean(), rewardIntakesEvery2500Episodes.sum()
        return allRewardIntakes, numberOfSteps


class AgentWithSingleAlgo(Agent):

    def __init__(self, maze, algo, params):
        self.maze = maze
        self.algo = algo(maze, params)
    
    def update(self, reward, new_state, action):
        self.algo.update(reward, new_state, action)

    def chooseAction(self):
        return self.algo.getActionBoltzmannChoice()


class AgentWithEnsemble(Agent):

    def __init__(self, maze, ensembleMethod, algoParamsList, temp, neural=False):
        self.maze = maze
        self.temp = temp
        self.ensembleMethod = ensembleMethod
        if not neural:
            self.algos = [
                QLearningNormal(maze, algoParamsList[0]),
                SARSANormal(maze, algoParamsList[1]),
                ActorCriticNormal(maze, algoParamsList[2]),
                QVLearningNormal(maze, algoParamsList[3]),
                ACLANormal(maze, algoParamsList[4])
            ]
        else:
            self.algos = [
                QLearningNeuronal(maze, algoParamsList[0]),
                SARSANeuronal(maze, algoParamsList[1]),
                ActorCriticNeuronal(maze, algoParamsList[2]),
                QVLearningNeuronal(maze, algoParamsList[3]),
                ACLANeuronal(maze, algoParamsList[4])
            ]

    def update(self, reward, new_state, action):
        for algo in self.algos:
            algo.update(reward, new_state, action)

    def chooseAction(self):
        return self.ensembleMethod(self.algos, self.temp)

####################################### EXPERIMENT 1 ###########################################

algosExp1 = [
    ("Q-Learning", QLearningNormal, AlgoParams(alpha=0.2, gamma=0.9, temp=1)),
    ("SARSA", SARSA, AlgoParams(alpha=0.2, gamma=0.9, temp=1)),
    ("Actor-Critic", ActorCritic, AlgoParams(alpha=0.1, beta=0.2, gamma=0.95, temp=1)),
    ("QV-Learning", QVLearning, AlgoParams(alpha=0.2, beta=0.2, gamma=0.9, temp=1)),
    ("ACLA", ACLA, AlgoParams(alpha=0.005, beta=0.1, gamma=0.99, temp=1/9))
]
algoParamsListExp1 = [param[2] for param in algosExp1]

ensemblesExp1 = [
    ("Majority", majorityVote, algoParamsListExp1, 1 / 1.6),
    ("Rank", rankVote, algoParamsListExp1, 1 / 0.6),
    ("Boltzmann Addition", boltzmannAddVote, algoParamsListExp1, 1 / 1),
    ("Boltzmann Multiplication", boltzmannMultVote, algoParamsListExp1, 1/ 0.2)
]

####################################### EXPERIMENT 3 ###########################################
algosExp3 = [
    ("Q-Learning", QLearningNeuronal, AlgoParams(alpha=0.01, gamma=0.95, temp=1, numHiddenNodes=60)),
    ("SARSA", SARSA, AlgoParams(alpha=0.01, gamma=0.95, temp=1, numHiddenNodes=60)),
    ("Actor-Critic", ActorCritic, AlgoParams(alpha=0.015, beta=0.003, gamma=0.95, temp=1, numHiddenNodes=60)),
    ("QV-Learning", QVLearning, AlgoParams(alpha=0.01, beta=0.01, gamma=0.9, temp=1/0.4, numHiddenNodes=60)),
    ("ACLA", ACLA, AlgoParams(alpha=0.06, beta=0.002, gamma=0.98, temp=1/6, numHiddenNodes=60))
]
algoParamsListExp3 = [param[2] for param in algosExp3]

ensemblesExp3 = [
    ("Majority", majorityVote, algoParamsListExp3, 1 / 2.6),
    ("Rank", rankVote, algoParamsListExp3, 1 / 0.8),
    ("Boltzmann Addition", boltzmannAddVote, algoParamsListExp3, 1 / 1),
    ("Boltzmann Multiplication", boltzmannMultVote, algoParamsListExp3, 1/ 0.2)
]
################################################################################################

if __name__ == "__main__":

    maze = createDynamicObstaclesMaze()
    maze.render()
    start = time.time()
    agent = AgentWithSingleAlgo(maze, QLearningNeuronal, algoParamsListExp3[0])
    rewards, numSteps = agent.learn(5000)
    print("Took %d s" % (time.time() - start))
    saveToFile("results/nnQLearningRewards.json", rewards)
    saveToFile("results/nnQLearningNumSteps.json", numSteps)


    # maze = createDynamicObstaclesMaze()
    # start = time.time()
    # agent = AgentWithEnsemble(maze, boltzmannMultVote, algoParamsListExp3, 1/ 0.2, neural=True)
    # results = agent.learn(50000)
    # print("Took %d s" % (time.time() - start))
    # print(results)
    # saveToFile("nnQLearning.json", results)


    # for name, ensembleMethod, algoParamsList, temp in ensembles[-2:]:
    #     print(name)
    #     start = time.time()
    #     #agent = AgentWithSingleAlgo(maze, ensembleMethod, algoParamsList, temp)
    #     #agent = AgentWithEnsemble(maze, ensembleMethod, algoParamsList, temp)
    #     agent.learn(50000)
    #     print("Took %d s" % (time.time() - start))

    # for name, algo, param in algos:
    #     maze = createSimpleMaze()
    #     agent = AgentWithSingleAlgo(maze, algo, param)

    #     print("Testing %s..." % name)

    #     start = time.time()

    #     # reward intake = reward moyen par mouvement
    #     # Il mesure deux choses
    #     # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    #     # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
    #     allRewardIntakes, numberOfSteps, final, cumulative = agent.learn(50000)
    #     print("Learning process took %d seconds" % (time.time() - start))
    #     print("Final: %s, Cumulative: %s" % (final, cumulative))

    #     saveToFile("results/data_%s.json" % name, [final, cumulative])
    #     saveToFile("results/allRewardsIntakes_%s.json" % name, allRewardIntakes)
    #     saveToFile("results/numberOfSteps_%s.json" % name, numberOfSteps)
    #     print()

    #name = "Majority"
    #maze = createSimpleMaze()
    #agent = AgentWithEnsemble(maze, majorityVote, [param[2] for param in algos], 1/1.6)
    #allRewardIntakes, numberOfSteps, final, cumulative = agent.learn(50000)
    #saveToFile("results/data_%s.json" % name, [final, cumulative])
    #saveToFile("results/allRewardsIntakes_%s.json" % name, allRewardIntakes)
    #saveToFile("results/numberOfSteps_%s.json" % name, numberOfSteps)
    pass
