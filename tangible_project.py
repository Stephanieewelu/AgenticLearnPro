import streamlit as st
from AgenticLearnPro.agent import QLearningAgent
from AgenticLearnPro.environment import SimpleEnv

st.title("Smart Delivery Robot Simulation")
st.write("This project simulates a smart delivery robot using Q-learning to find the optimal path in a warehouse environment. Powered by AgenticLearnPro.")

# Define a simple warehouse environment (grid)
grid_size = st.slider("Warehouse grid size", 4, 10, 6)
start = (0, 0)
goal = (grid_size-1, grid_size-1)

# Custom environment for grid navigation
class GridEnv(SimpleEnv):
    def __init__(self, size, start, goal):
        self.size = size
        self.start = start
        self.goal = goal
        self.state_space = [(i, j) for i in range(size) for j in range(size)]
        self.action_space = ['up', 'down', 'left', 'right']
        self.reset()
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
    def step(self, action):
        x, y = self.agent_pos
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.size - 1:
            y += 1
        self.agent_pos = (x, y)
        reward = 1 if self.agent_pos == self.goal else -0.1
        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

env = GridEnv(grid_size, start, goal)
agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

# Train the agent
episodes = st.slider("Training episodes", 10, 500, 100)
progress = st.progress(0)
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
    agent.decay_exploration()
    progress.progress((episode + 1) / episodes)

st.success(f"Training completed for {episodes} episodes!")

# Test the trained agent
state = env.reset()
done = False
steps = [state]
while not done:
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    steps.append(next_state)
    state = next_state

st.subheader("Robot Path from Start to Goal")
warehouse = [['⬜' for _ in range(grid_size)] for _ in range(grid_size)]
for (i, j) in steps:
    warehouse[i][j] = '🤖'
warehouse[start[0]][start[1]] = '🚩'
warehouse[goal[0]][goal[1]] = '🏁'
for row in warehouse:
    st.write(' '.join(row))

st.info("This tangible project demonstrates how AgenticLearnPro can power smart robotics and automation use cases.")