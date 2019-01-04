import numpy as np
import random

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
REWARD_MOVE = -0.1

ALL_POSITIONS = [(x,y) for x in range(WIDTH) for y in range(HEIGHT)]

class Maze:

    def __init__(self, start, goal, obstacles):

        self.start = start
        self.goal = goal
        self.obstacles = obstacles

        self.agent_position = self.start
        self.actions_counter = 0

    def getActions(self):
        return [0, 1, 2, 3]

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

    def neighbors(self, pos):
        """
        Returns all the neighbors of a given position: left, right, up and down.
        These must be valid: in the grid and not obstacles.
        """
        x, y = pos
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbors if not self.isOutOfBounds(n) and n not in self.obstacles]


    def hasSolution(self):
        """ 
        Returns whether there is a path from the start to the goal.
        """
        toExplore = set([self.start])
        explored = set([])
        while toExplore:
            pos = toExplore.pop()
            if pos == self.goal:
                return True

            explored.add(pos)

            for neighbor in self.neighbors(pos):
                if neighbor not in explored:
                    toExplore.add(neighbor)
        
        return False

    def isObstacle(position):
        return position in self.obstacles

    def render(self):
        print("-" * (WIDTH + 2))
        for y in range(HEIGHT - 1, -1, -1):
            print('|', end="")
            for x in range(WIDTH):
                if self.agent_position == (x, y):
                    print('A', end="")
                elif (x, y) in self.obstacles:
                    print('X', end="")
                elif (x, y) == self.goal:
                    print('G', end="")
                else:
                    print(' ', end="")
            print('|')
        print("-" * (WIDTH + 2))

    def getObservation(self):
        observation = np.zeros(4)
        x, y = self.agent_position
        for i, adjPos in enumerate(allNeighbors(x, y)):
            if self.isOutOfBounds(adjPos) or adjPos in self.obstacles:
                observation[i] = 1
            else:
                observation[i] = 0

            if np.random.random() < 0.1:
                observation[i] = 0 if observation[i] == 1 else 0

        return observation



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
    numObstacles = np.random.choice([4, 5, 6, 7, 8])
    start = (0, 3)
    goal = (8, 5)

    availablePos = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)
                    if (x, y) != start and (x, y) != goal]

    while True:
        # sample returns a k-element sample from the population (no replacement)
        obstacles = random.sample(availablePos, numObstacles)
        maze = Maze(start, goal, obstacles)
        if maze.hasSolution():
            return maze



def createGeneralizedMaze():
    """
    Creates a Maze with a random goal and between 4 and 8 obstacles placed 
    at random positions 
    """
    numObstacles = np.random.choice([4, 5, 6, 7, 8])
    start = (0, 3)
    availablePos = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)
                    if (x, y) != start]

    while True:
        # sample returns a k-element sample from the population (no replacement)
        positions = random.sample(availablePos, numObstacles + 1)
        goal, obstacles = positions[0], positions[1:]
        maze = Maze(start, goal, obstacles)
        if maze.hasSolution():
            return maze

##########################
####   Belief State   ####
##########################


def initBeliefState(obstacles):
    beliefState = np.random.uniform(size=(WIDTH, HEIGHT))
    for obstacle in obstacles:
        beliefState[obstacle] = 0
    return beliefState

def manathanDistance(s, newS):
    x, y, newX, newY = s, newS
    return abs(newX - x) + abs(newY -y)


def allNeighbors(x, y):
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def T(s, a, newS):
    sX, sY = s
    deltaX, deltaY = a
    newX, newY = sX + deltaX, sY + deltaY

    if manathanDistance(s, newS) > 1:
        return 0
    elif s == newS:
        res = 0
        for neighbor in allNeighbors(sX, sY):
            if isObstacle(neighbor) or isOutOfBound(neighbor):
                if neighbor == (newX, newY):
                    res += 0.8     # 80.0%
                else:
                    res += 0.2 / 3 # 6.66%

        return res
    elif (newX, newY) == newS:
        return 0.8
    else:
        return 0.2

def getProbabilityOfSeeingWallObservationAt(seenWall, x, y):
    if (isWall(x, y) or isOutOfBounds(x, y)) == seenWall: # if seen correctly
        return 0.9
    else:
        return 0.1

def P(observation, s):
    x, y = s
    res = 1
    for wallObs in observation:
        res *= getProbabilityOfSeeingWallObservationAt(wallObs, x, y)
    return res
             
    
def possiblePositions(x, y):
    return allNeighbors(x, y) + [(x, y)]

def normalisation(obs, beliefState, action):
    p = 0
    for newS in ALL_POSITIONS:
        for oldS in ALL_POSITIONS:
            p += P(obs, newS) * T(oldS, action, newS) * beliefState[oldS]

    return 1 / p

def updateB(beliefState, newS, obs, action):
    # Compute new value
    newX, newY = newS
    res = 0
    for oldS in possiblePositions(newX, newY):
        res += T(oldS, action, newS) * beliefState[oldS]
    res *= P(obs, newS)
    res *= normalisation(obs, beliefState, action)
    
    # Update value in matrix
    beliefState[newS] = res








if __name__ == "__main__":
    env = createSimpleMaze()
    # print(reward, new_state)
    env.render()
