import numpy as np

class SimpleEnv:
    def __init__(self):
        self.state_space = 5  # 5 possible states
        self.action_space = 2  # 2 actions: 0 (left), 1 (right)
        self.state = 0  # Start at state 0
        self.goal = 4   # Goal is state 4

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Action 0: move left, Action 1: move right
        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 1 and self.state < self.state_space - 1:
            self.state += 1

        # Reward: +10 for reaching the goal, -1 otherwise
        reward = 10 if self.state == self.goal else -1
        done = self.state == self.goal
        return self.state, reward, done