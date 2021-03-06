from algos import *
from ensembleMethods import *
from mazes import *
from helper import *
import time





class Agent():

    def update(self, reward, new_state, action):
        raise NotImplementedError

    def chooseAction(self):
        raise NotImplementedError

    def learn(self, episodes):
        delay = int(episodes // 20)
        rewardIntakesOverEpisode = np.zeros(20)
        lastRewardIntakes = np.zeros(int(delay))

        for episodeNum in range(episodes):
            episodeReward = 0
            while not self.maze.isDone():
                action = self.chooseAction()
                reward, new_state = self.maze.step(action)
                episodeReward += reward
                # maze.render()
                self.update(reward, new_state, action)

            #allRewardIntakes[episodeNum] = (episodeReward / self.maze.actions_counter)
            #numberOfSteps[episodeNum] = self.maze.actions_counter
            # if episodeNum % 100 == 0 : 
                #print(episodeNum, self.maze.actions_counter)

            if (episodeNum % delay) == (delay - 1):
                # print("Done %d " % (episodeNum + 1))
                rewardIntakesOverEpisode[int(episodeNum // delay)] = (episodeReward / self.maze.actions_counter)

            if episodeNum + delay >= episodes:
                lastRewardIntakes[episodeNum - (episodes - int(delay))] = (episodeReward / self.maze.actions_counter)

            self.maze.reset()

        #return last2500RewardIntakes.mean(), rewardIntakesEvery2500Episodes.sum()
        #return allRewardIntakes, numberOfSteps, last2500RewardIntakes.mean(), rewardIntakesEvery2500Episodes.sum()
        #return allRewardIntakes, numberOfSteps
        return lastRewardIntakes, rewardIntakesOverEpisode


class AgentWithSingleAlgo(Agent):

    def __init__(self, maze, algo, params, beliefState):
        self.maze = maze
        self.algo = algo(maze, params, beliefState)
    
    def update(self, reward, new_state, action):
        obs = self.maze.getObservation() if self.algo.beliefState is not None else None
        self.algo.update(reward, new_state, action, obs)

    def chooseAction(self):
        return self.algo.getActionBoltzmannChoice()


class AgentWithEnsemble(Agent):

    def __init__(self, maze, ensembleMethod, algoParamsList, temp, beliefState, neural=False):
        self.maze = maze
        self.temp = temp
        self.ensembleMethod = ensembleMethod
        if not neural:
            self.algos = [
                QLearningNormal(maze, algoParamsList[0], None),
                SARSANormal(maze, algoParamsList[1], None),
                ActorCriticNormal(maze, algoParamsList[2], None),
                QVLearningNormal(maze, algoParamsList[3], None),
                ACLANormal(maze, algoParamsList[4], None)
            ]
        else:
            self.algos = [
                QLearningNeuronal(maze, algoParamsList[0], beliefState),
                SARSANeuronal(maze, algoParamsList[1], beliefState),
                ActorCriticNeuronal(maze, algoParamsList[2], beliefState),
                QVLearningNeuronal(maze, algoParamsList[3], beliefState),
                ACLANeuronal(maze, algoParamsList[4], beliefState)
            ]

    def update(self, reward, new_state, action):
        obs = self.maze.getObservation() if self.algos[0].beliefState is not None else None
        for algo in self.algos:
            algo.update(reward, new_state, action, obs)

    def chooseAction(self):
        return self.ensembleMethod(self.algos, self.temp)

if __name__ == "__main__":

    from experimentConf import *
    maze = createSimpleMaze()
    maze.render()
    start = time.time()
    agent = AgentWithEnsemble(maze, rankVote, algoParamsListExp1, 1 / 0.6, None)
    rewards, numSteps = agent.learn(20000)
    # print("Took %d s" % (time.time() - start))
    # saveToFile("results/nnACLARewards.json", rewards)
    # saveToFile("results/nnACLANumSteps.json", numSteps)


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
