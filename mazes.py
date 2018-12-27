import numpy as np

WIDTH = 9
HEIGHT = 6

ACTIONS = {
    0: (0, 1),
    1: (0, -1),
    2: (1, 0),
    3: (-1, 0)
}

PENALTY = -2
REWARD_GOAL = 100
REWARD_MOVE = -1


class Maze:

    def __init__(self, start, goal, obstacles):

        self.start = start
        self.goal = goal
        self.obstacles = obstacles

        self.agent_position = self.start
        self.actions_counter = 0

    def getActions(self):
        return (list(ACTIONS.keys()))

    def reset(self):
        self.agent_position = self.start
        self.actions_counter = 0

    def isOutOfBounds(self, position):
        """
        Checks if a position is outside of the maze bounds
        """
        x, y = position
        return x >= WIDTH or x < 0 or y >= HEIGHT or y < 0

    def apply_noise(self, action):
        """
        Replaces an action by another one
        with a probability of 20%
        """
        if np.random.random() < 0.2:
            return np.random.choice([new_action for new_action in self.getActions() if new_action != action])

        return action

    def step(self, action):
        """
        Makes the agent move.
        Returns the reward his new position
        """

        action = self.apply_noise(action)

        new_pos_x = self.agent_position[0] + ACTIONS[action][0]
        new_pos_y = self.agent_position[1] + ACTIONS[action][1]

        if (new_pos_x, new_pos_y) == self.goal:
            self.agent_position = (new_pos_x, new_pos_y)
            reward = REWARD_GOAL

        elif ((new_pos_x, new_pos_y) in self.obstacles) or (self.isOutOfBounds((new_pos_x, new_pos_y))):
            self.agent_position = self.agent_position
            reward = PENALTY

        else:
            self.agent_position = (new_pos_x, new_pos_y)
            reward = REWARD_MOVE

        self.actions_counter += 1
        return reward, self.agent_position

    def isDone(self):
        """
        Checks if the agent has reached the goal
        """
        return self.agent_position == self.goal or self.actions_counter == 1000

    def render(self):
        for y in range(HEIGHT - 1, -1, -1):
            for x in range(WIDTH):
                if self.agent_position == (x, y):
                    print('A', end=" ")
                elif (x, y) in self.obstacles:
                    print('X', end=" ")
                elif (x, y) == self.goal:
                    print('G', end=" ")
                else:
                    print('O', end=" ")
            print('\n')


#########################
#### Maze generators ####
#########################

def createSimpleMaze():
    """
    Creates the "default" maze.
    Used for the Experiment 1
    """
    start = (0, 3)
    goal = (8, 5)
    obstacles = [(2, 2), (2, 3), (2, 4), (5, 1),
                 (7, 3), (7, 4), (7, 5)]
    return Maze(start, goal, obstacles)


def createDynamicGoalMaze():
    """
    Creates a maze with the goal placed at a random position.
    Used for the Experiment 2
    """
    start = (0, 3)
    obstacles = [(2, 2), (2, 3), (2, 4), (5, 1),
                 (7, 3), (7, 4), (7, 5)]

    possibleGoalPos = [(x, y) for x in range(WIDTH) for y in range(
        HEIGHT) if (x, y) != start and (x, y) not in obstacles]
    goal = possibleGoalPos[np.random.randint(len(possibleGoalPos) - 1)]

    return Maze(start, goal, obstacles)


def createDynamicObstaclesMaze():
    """
    Creates a Maze with between 4 and 8 obstacles placed 
    at random positions
    """
    # TODO
    pass


def createGeneralizedMaze():
    # TODO
    pass


if __name__ == "__main__":
    env = createDynamicGoalMaze()
    # print(reward, new_state)
    env.render()
