import numpy as np

class QLearningAgent:
    """
    A Q-learning agent for reinforcement learning.

    This class implements a Q-learning algorithm to train an agent in a given environment.
    It maintains a Q-table to store action values and supports exploration-exploitation trade-off.

    Args:
        state_space (int): Number of possible states in the environment.
        action_space (int): Number of possible actions in the environment.
        learning_rate (float, optional): Learning rate for Q-value updates. Defaults to 0.1.
        discount_factor (float, optional): Discount factor for future rewards. Defaults to 0.9.
        exploration_rate (float, optional): Initial rate for exploration. Defaults to 1.0.

    Attributes:
        q_table (ndarray): Q-value table for state-action pairs.
        learning_rate (float): Learning rate for updates.
        discount_factor (float): Discount factor for future rewards.
        exploration_rate (float): Current exploration rate.
        action_space (int): Number of possible actions.
    """

    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.q_table = np.zeros((state_space, action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.action_space = action_space

    def choose_action(self, state):
        """
        Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (int): Current state index.

        Returns:
            int: Selected action index.
        """
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_space)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        """
        Update the Q-table based on the Q-learning update rule.

        Args:
            state (int): Current state index.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Next state index.
        """
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state, action] = new_q

    def decay_exploration(self, decay_rate=0.995, min_rate=0.1):
        """
        Decay the exploration rate over time.

        Args:
            decay_rate (float, optional): Rate to decay exploration. Defaults to 0.995.
            min_rate (float, optional): Minimum exploration rate. Defaults to 0.1.
        """
        self.exploration_rate = max(min_rate, self.exploration_rate * decay_rate)

def main():
    """
    Main function to train the Q-learning agent.

    This function initializes a SimpleEnv environment, trains the QLearningAgent
    for 100 episodes, and prints the final Q-table.

    Returns:
        None
    """
    from environment import SimpleEnv

    env = SimpleEnv()
    agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

    for episode in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        agent.decay_exploration()
    print("Q-table after training:", agent.q_table)

if __name__ == "__main__":
    main()