import numpy as np

class QLearningAgent:
    """
    A Q-learning agent for reinforcement learning.

    This class implements a Q-learning algorithm to train an agent in a given environment.
    It maintains a Q-table to store action values and supports exploration-exploitation trade-off.

    Args:
        state_space (int or list): Number of possible states or list of possible states in the environment.
        action_space (int or list): Number of possible actions or list of possible actions in the environment.
        learning_rate (float, optional): Learning rate for Q-value updates. Defaults to 0.1.
        discount_factor (float, optional): Discount factor for future rewards. Defaults to 0.9.
        exploration_rate (float, optional): Initial rate for exploration. Defaults to 1.0.

    Attributes:
        q_table (dict or ndarray): Q-value table for state-action pairs.
        learning_rate (float): Learning rate for updates.
        discount_factor (float): Discount factor for future rewards.
        exploration_rate (float): Current exploration rate.
        action_space (int or list): Number or list of possible actions.
    """

    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.action_space = action_space
        
        # Handle different types of state_space and action_space
        if isinstance(state_space, int) and isinstance(action_space, int):
            # Original behavior for integer inputs
            self.q_table = np.zeros((state_space, action_space))
        else:
            # Dictionary-based Q-table for non-integer state/action spaces
            self.q_table = {}
            self.state_space = state_space
            
            # Initialize Q-table with zeros
            if isinstance(state_space, list):
                for state in state_space:
                    self.q_table[state] = {}
                    for action in action_space:
                        self.q_table[state][action] = 0.0

    def _ensure_state_exists(self, state):
        """
        Ensure that a state exists in the Q-table.
        
        Args:
            state: The state to check/add.
        """
        if isinstance(self.q_table, dict) and state not in self.q_table:
            self.q_table[state] = {}
            for action in self.action_space:
                self.q_table[state][action] = 0.0

    def choose_action(self, state, explore=True):
        """
        Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (int or any): Current state index or state representation.
            explore (bool, optional): Whether to use exploration or not. Defaults to True.

        Returns:
            int or any: Selected action index or action representation.
        """
        # Check if we're using the dictionary-based Q-table
        if isinstance(self.q_table, dict):
            # Initialize state if not in Q-table
            self._ensure_state_exists(state)
                    
            if explore and np.random.uniform(0, 1) < self.exploration_rate:
                return np.random.choice(self.action_space)  # Explore
            else:
                # Find action with maximum Q-value
                return max(self.q_table[state].items(), key=lambda x: x[1])[0]  # Exploit
        else:
            # Original behavior for ndarray Q-table
            if explore and np.random.uniform(0, 1) < self.exploration_rate:
                return np.random.choice(self.action_space)  # Explore
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        """
        Update the Q-table based on the Q-learning update rule.

        Args:
            state (int or any): Current state index or state representation.
            action (int or any): Action taken in current state.
            reward (float): Reward received after taking the action.
            next_state (int or any): Next state index or state representation.
        """
        # Check if we're using the dictionary-based Q-table
        if isinstance(self.q_table, dict):
            # Initialize states if not in Q-table
            self._ensure_state_exists(state)
            self._ensure_state_exists(next_state)
            
            # Get the current Q-value
            current_q = self.q_table[state][action]
            
            # Find the maximum Q-value for the next state
            best_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            
            # Calculate the new Q-value
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
            
            # Update the Q-table
            self.q_table[state][action] = new_q
        else:
            # Original behavior for ndarray Q-table
            current_q = self.q_table[state, action]
            best_next_q = np.max(self.q_table[next_state])
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
            self.q_table[state, action] = new_q

    def decay_exploration(self, decay_rate=0.99):
        """
        Decay the exploration rate to gradually shift from exploration to exploitation.

        Args:
            decay_rate (float, optional): Rate at which to decay exploration. Defaults to 0.99.
        """
        self.exploration_rate *= decay_rate

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