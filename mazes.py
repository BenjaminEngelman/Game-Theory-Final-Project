import numpy as np

ACTIONS = {
    0: (0, 1),
    1: (0, -1),
    2: (1, 0),
    3: (-1, 0)
}

PENALTY = -2
REWARD_GOAL = 100
REWARD_MOVE = -1


class SimpleMaze:

    def __init__(self):
        self.WIDTH = 9
        self.HEIGHT = 6

        self.start = (0, 3)
        self.goal = (8, 5)
        self.obstacles = [(2, 2), (2, 3), (2, 4), (5, 1),
                          (7, 3), (7, 4), (7, 5)]

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
        return x >= self.WIDTH or x < 0 or y >= self.HEIGHT or y < 0

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
        for y in range(self.HEIGHT - 1, -1, -1):
            for x in range(self.WIDTH):
                if self.agent_position == (x, y): 
                    print('A', end=" ")
                    pass
                elif (x, y) in self.obstacles:
                    print('X', end=" ")
                    pass
                else:
                    print('O', end=" ")
                    pass
            print('\n')


if __name__ == "__main__":
    env = SimpleMaze()
    reward, new_state = env.step(0)
    # print(reward, new_state)
    env.render()