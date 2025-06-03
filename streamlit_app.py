import streamlit as st
from AgenticLearnPro.agent import QLearningAgent
from AgenticLearnPro.environment import SimpleEnv

st.title("AgenticLearnPro Q-Learning Demo")
st.write("This demo shows a Q-learning agent interacting with a simple environment using the AgenticLearnPro package.")

# Parameters
episodes = st.slider("Number of training episodes", 10, 500, 100)

# Create environment and agent
env = SimpleEnv()
agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

# Train the agent
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
total_reward = 0
steps = []
while not done:
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    steps.append((state, action, reward))
    total_reward += reward
    state = next_state

st.subheader("Test Run Results")
st.write(f"Total reward: {total_reward}")
st.table([
    {"State": s, "Action": a, "Reward": r} for s, a, r in steps
])

st.info("Fork this repo, install requirements, and run `streamlit run streamlit_app.py` to try it locally!")