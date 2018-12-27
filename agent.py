from algos import *
from ensembleMethods import *
from mazes import SimpleMaze


class Agent():

    def __init__(self, maze, ensembleMethod):
        self.maze = maze
        self.ensembleMethod = ensembleMethod
        self.algos = [
            QLearning(maze),
            SARSA(maze),
            ActorCritic(maze),
            QVLearning(maze),
            ACLA(maze)
        ]

    def learn(self, episodes):
        rewardsOverTime = []

        for _ in range(episodes):
            episodeReward = 0
            while not self.maze.isDone():
                action = self.ensembleMethod(self.algos)
                reward, new_state = self.maze.step(action)
                episdodeReward += reward

                for algo in self.algos:
                    algo.update(reward, new_state, action)

            rewardsOverTime.append(episodeReward)

        return rewardsOverTime


if __name__ == "__main__":
    maze = SimpleMaze()
    agent = Agent(SimpleMaze, "METHOD")
    rewardsOverTime = agent.learn(50000)
