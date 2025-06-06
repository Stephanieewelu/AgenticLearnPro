from AgenticLearnPro.agent import QLearningAgent
from AgenticLearnPro.environment import SimpleEnv

# Create environment and agent
env = SimpleEnv()
agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

# Train the agent
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
    agent.decay_exploration()

# Test the trained agent (disable exploration for testing)
agent.exploration_rate = 0  # Use the best actions
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    total_reward += reward
    state = next_state
    print(f"State: {state}, Action: {action}, Reward: {reward}")
print(f"Total reward: {total_reward}")