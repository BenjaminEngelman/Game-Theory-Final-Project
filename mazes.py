ACTIONS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0)
}

PENALTY = -2
REWARD_GOAL = 100
REWARD_MOVE = -1


class SimpleMaze:

    def __init__(self):
        self.WIDTH = 9
        self.HEIGHT = 6

        self.start = (0, 3)
        self.goal = (8, 7)
        self.obstacles = [(2, 2), (2, 3), (2, 4), (5, 1),
                          (7, 5), (7, 6), (7, 7)]

        self.agent_position = self.start

    def getActions(self):
        return (list(ACTIONS.keys()))

    def isOutOfBounds(self, position):
        """
        Checks if a position is outside of the maze bounds
        """
        x, y = position
        return x >= self.WIDTH or x < 0 or y >= self.HEIGHT or y < 0

    def step(self, action):
        """
        Makes the agent move.
        Returns the reward his new position
        """
        new_pos_x = self.agent_position[0] + ACTIONS[action][0]
        new_pos_y = self.agent_position[1] + ACTIONS[action][1]

        if self.isDone():
            self.agent_position = (new_pos_x, new_pos_y)
            reward = REWARD_GOAL

        elif ((new_pos_x, new_pos_y) in self.obstacles) or (self.isOutOfBounds((new_pos_x, new_pos_y))):
            self.agent_position = self.agent_position
            reward = PENALTY

        else:
            self.agent_position = (new_pos_x, new_pos_y)
            reward = REWARD_MOVE

        return reward, self.agent_position

    def isDone(self):
        """
        Checks if the agent has reached the goal
        """
        return self.agent_position == self.goal


if __name__ == "__main__":
    env = SimpleMaze()
    reward, new_state = env.step("W")
    print(reward, new_state)
