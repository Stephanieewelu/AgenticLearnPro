import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AgenticLearnPro.agent import QLearningAgent
from AgenticLearnPro.environment import SimpleEnv

st.title("AI Stock Trading Agent")
st.write("This application demonstrates how AgenticLearnPro can be used to create an AI agent that learns to trade stocks.")

# Custom environment for stock trading
class StockTradingEnv(SimpleEnv):
    def __init__(self, data):
        self.data = data
        self.state_space = list(range(len(data) - 1))
        self.action_space = ['buy', 'sell', 'hold']
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.portfolio = {'cash': 10000, 'shares': 0}
        self.trades = []
        return self.current_step
    
    def step(self, action):
        # Get current price
        current_price = self.data.iloc[self.current_step]['price']
        next_price = self.data.iloc[self.current_step + 1]['price']
        
        # Execute action
        if action == 'buy' and self.portfolio['cash'] >= current_price:
            # Buy as many shares as possible
            shares_to_buy = self.portfolio['cash'] // current_price
            self.portfolio['shares'] += shares_to_buy
            self.portfolio['cash'] -= shares_to_buy * current_price
            self.trades.append(('buy', self.current_step, shares_to_buy, current_price))
        
        elif action == 'sell' and self.portfolio['shares'] > 0:
            # Sell all shares
            self.portfolio['cash'] += self.portfolio['shares'] * current_price
            self.trades.append(('sell', self.current_step, self.portfolio['shares'], current_price))
            self.portfolio['shares'] = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        portfolio_value_before = self.portfolio['cash'] + self.portfolio['shares'] * current_price
        portfolio_value_after = self.portfolio['cash'] + self.portfolio['shares'] * next_price
        reward = portfolio_value_after - portfolio_value_before
        
        # Check if done
        done = self.current_step >= len(self.data) - 2
        
        return self.current_step, reward, done
    
    def get_portfolio_value(self):
        current_price = self.data.iloc[self.current_step]['price']
        return self.portfolio['cash'] + self.portfolio['shares'] * current_price

# Generate sample stock data
def generate_stock_data(days=100, volatility=0.01):
    np.random.seed(42)  # For reproducibility
    price = 100
    prices = [price]
    
    for _ in range(days - 1):
        change = np.random.normal(0, volatility)
        price *= (1 + change)
        prices.append(price)
    
    dates = pd.date_range(start='2023-01-01', periods=days)
    return pd.DataFrame({'date': dates, 'price': prices})

# App functionality
st.sidebar.header("Simulation Parameters")
days = st.sidebar.slider("Number of trading days", 50, 200, 100)
volatility = st.sidebar.slider("Market volatility", 0.01, 0.05, 0.02)
training_episodes = st.sidebar.slider("Training episodes", 10, 500, 100)

# Generate data
stock_data = generate_stock_data(days, volatility)

# Display stock chart
st.subheader("Stock Price Chart")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(stock_data['date'], stock_data['price'])
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
st.pyplot(fig)

# Create environment and agent
env = StockTradingEnv(stock_data)
agent = QLearningAgent(state_space=env.state_space, action_space=env.action_space)

# Train the agent
if st.button("Train Trading Agent"):
    progress = st.progress(0)
    rewards_history = []
    
    for episode in range(training_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        agent.decay_exploration()
        progress.progress((episode + 1) / training_episodes)
    
    st.success(f"Training completed for {training_episodes} episodes!")
    
    # Plot training rewards
    st.subheader("Training Rewards")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards_history)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    st.pyplot(fig)
    
    # Test the trained agent
    state = env.reset()
    done = False
    portfolio_values = [env.get_portfolio_value()]
    actions_taken = []
    
    while not done:
        action = agent.choose_action(state)
        actions_taken.append(action)
        next_state, reward, done = env.step(action)
        state = next_state
        portfolio_values.append(env.get_portfolio_value())
    
    # Display results
    st.subheader("Trading Results")
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    roi = (final_value - initial_value) / initial_value * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Portfolio", f"${initial_value:.2f}")
    col2.metric("Final Portfolio", f"${final_value:.2f}")
    col3.metric("Return on Investment", f"{roi:.2f}%")
    
    # Plot portfolio value over time
    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(portfolio_values)), portfolio_values)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Portfolio Value ($)')
    st.pyplot(fig)
    
    # Display trades
    if env.trades:
        st.subheader("Trading Activity")
        trades_df = pd.DataFrame(env.trades, columns=['action', 'day', 'shares', 'price'])
        trades_df['value'] = trades_df['shares'] * trades_df['price']
        st.dataframe(trades_df)

st.info("""
### How to Use This Project

1. Adjust the simulation parameters in the sidebar
2. Click 'Train Trading Agent' to start the simulation
3. Review the agent's performance and trading decisions

This project demonstrates how AgenticLearnPro can be used to create intelligent trading systems that learn optimal strategies through reinforcement learning.
""")