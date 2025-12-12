import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import copy
from collections import deque
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# PART 1: THE FINANCIAL ENVIRONMENT (WHERE YOU FEED YOUR DATA)
# ----------------------------------------------------------------------------
# This class turns your timeseries data (e.g., a CSV) into a "game"
# that the RL agent can play and learn from.

class FinancialEnv:
    """
    A simple Reinforcement Learning environment for financial timeseries data.
    
    This is the key component where you "feed in data".
    You initialize this class with a pandas DataFrame.
    """
    def __init__(self, data, window_size=10):
        """
        Initializes the environment.
        
        Args:
            data (pd.DataFrame): A DataFrame with your financial data. 
                                 Must include a 'price' column.
            window_size (int): How many past timesteps to include in the state.
                               This creates the "sliding window" vector.
        """
        self.data = data
        self.window_size = window_size
        self.prices = data['price'].values
        self.features = data.drop(columns=['price']).values
        
        # Define state and action space
        # State: (window_size * num_features) + (window_size * prices)
        self.state_dim = (window_size * self.features.shape[1]) + window_size
        self.action_dim = 3  # 0: Hold, 1: Buy, 2: Sell
        
        self.reset()

    def _get_state(self):
        """
        Generates the state vector for the current timestep.
        The state is a flattened vector of the past 'window_size' days
        of features and prices.
        """
        if self.current_step < self.window_size - 1:
            # Not enough history, pad with zeros
            features = np.zeros((self.window_size, self.features.shape[1]))
            prices = np.zeros(self.window_size)
            
            features[-(self.current_step+1):] = self.features[:self.current_step+1]
            prices[-(self.current_step+1):] = self.prices[:self.current_step+1]
        else:
            # Full window
            start = self.current_step - (self.window_size - 1)
            end = self.current_step + 1
            features = self.features[start:end]
            prices = self.prices[start:end]
        
        # Flatten the window into a single 1D vector
        return np.concatenate([features.flatten(), prices.flatten()])

    def reset(self):
        """Resets the environment to the beginning of the data."""
        self.current_step = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        Takes an action and moves the environment one step forward.
        
        Args:
            action (int): 0 (Hold), 1 (Buy), or 2 (Sell)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        if self.done:
            raise Exception("Episode is done, please call reset().")

        self.current_step += 1
        
        # Check if the episode is over
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            reward = 0
            next_state = self._get_state()
            return next_state, reward, self.done

        # --- Calculate Reward ---
        # This is a simple reward logic. A real one would be more complex.
        current_price = self.prices[self.current_step - 1]
        next_price = self.prices[self.current_step]
        
        if action == 1:  # Buy
            reward = next_price - current_price
        elif action == 2:  # Sell
            reward = current_price - next_price
        else:  # Hold (action == 0)
            reward = 0
            
        next_state = self._get_state()
        
        return next_state, reward, self.done

# ----------------------------------------------------------------------------
# PART 2: THE PROSPECT THEORY AGENT (THE "MODEL")
# ----------------------------------------------------------------------------
# This is the "brain" you designed, which combines the Utility
# and Probability layers into a single decision network.

class UtilityLayer(nn.Module):
    """
    Implements the Prospect Theory Value Function v(x).
    It learns the subjective utility of an objective outcome (reward).
    """
    def __init__(self, input_size=1, hidden_size=32):
        super(UtilityLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x is a tensor of potential outcomes (rewards)
        # We need to reshape for the linear layer if it's not already
        is_batch_of_outcomes = x.dim() > 1
        
        if not is_batch_of_outcomes:
            x = x.unsqueeze(-1) # Add feature dim: [batch] -> [batch, 1]
            
        h = F.relu(self.fc1(x))
        subjective_value = self.fc2(h) # This learns v(x)
        
        if not is_batch_of_outcomes:
            subjective_value = subjective_value.squeeze(-1) # [batch, 1] -> [batch]
            
        return subjective_value

class ProbabilityWeightingLayer(nn.Module):
    """
    Implements the Prospect Theory Weighting Function w(p).
    It learns the subjective weight of an objective probability.
    """
    def __init__(self, input_size=1, hidden_size=32):
        super(ProbabilityWeightingLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, p):
        # p is a tensor of probabilities
        h = F.relu(self.fc1(p.unsqueeze(-1)))
        # Output must be a positive weight. Sigmoid bounds it (0, 1).
        decision_weight = torch.sigmoid(self.fc2(h)) 
        return decision_weight.squeeze(-1)

class ProspectTheoryQNetwork(nn.Module):
    """
    The main RL agent (DQN) that integrates the two layers.
    This is a "model-based" agent, as it contains a 'world_model'
    to predict the distribution of future outcomes.
    
    *** NOTE: Model-based RL is notoriously hard to train. ***
    *** This 'world_model' has to learn the market's dynamics ***
    *** from scratch, which is a very complex task. ***
    """
    def __init__(self, state_dim, action_dim, num_outcomes=5, hidden_dim=128):
        super(ProspectTheoryQNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_outcomes = num_outcomes
        
        # --- These are your two custom layers ---
        self.utility_layer = UtilityLayer()
        self.probability_layer = ProbabilityWeightingLayer()
        
        # --- This network IS the 'world_model' ---
        # It predicts a distribution of (K) outcomes and their probabilities
        # for a given (state, action) pair.
        self.world_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output: K outcomes + K probability_logits
            nn.Linear(hidden_dim, num_outcomes * 2) 
        )

    def forward(self, state):
        """
        Calculates the Subjective Q-Value (SQ-Value) for all
        possible actions given a batch of states.
        
        Output: A tensor of [batch_size, action_dim]
                e.g., [[SQ_hold, SQ_buy, SQ_sell],  (for state 1)
                       [SQ_hold, SQ_buy, SQ_sell]]  (for state 2)
        """
        batch_size = state.shape[0]
        
        # 1. Expand state to evaluate all actions
        # [batch_size, state_dim] -> [batch_size * action_dim, state_dim]
        expanded_state = state.repeat_interleave(self.action_dim, dim=0)
        
        # 2. Create one-hot action tensors
        # [batch_size * action_dim, action_dim]
        actions_one_hot = torch.eye(self.action_dim).repeat(batch_size, 1)
        actions_one_hot = actions_one_hot.to(state.device)

        # 3. Concatenate state and action
        state_action_input = torch.cat([expanded_state, actions_one_hot], dim=1)
        
        # 4. Predict Outcome Distributions from the 'world_model'
        dist_output = self.world_model(state_action_input)
        
        # 5. Separate outcomes and probabilities
        # [batch_size * action_dim, num_outcomes]
        outcomes = dist_output[..., :self.num_outcomes]
        probs_logits = dist_output[..., self.num_outcomes:]
        
        # Use softmax to get valid probabilities
        # [batch_size * action_dim, num_outcomes]
        probs = F.softmax(probs_logits, dim=-1)

        # 6. Apply Prospect Theory Layers
        
        # v(x): Apply Utility Layer to outcomes
        # Reshape to [batch * action * num_outcomes, 1] to apply v(x) to each outcome
        outcomes_flat = outcomes.reshape(-1, 1)
        subjective_values_flat = self.utility_layer(outcomes_flat)
        subjective_values = subjective_values_flat.view(outcomes.shape)
        
        # w(p): Apply Probability Layer to probabilities
        # Reshape to [batch * action * num_outcomes, 1]
        probs_flat = probs.reshape(-1, 1)
        decision_weights_flat = self.probability_layer(probs_flat)
        decision_weights = decision_weights_flat.view(probs.shape)
        
        # 7. Integrate into Final Subjective Q-Value (SQ)
        # SQ = Σ [ w(p) * v(x) ]
        sq_values = torch.sum(decision_weights * subjective_values, dim=-1)
        
        # 8. Reshape to [batch_size, action_dim]
        final_q_values = sq_values.view(batch_size, self.action_dim)
        
        return final_q_values

# ----------------------------------------------------------------------------
# PART 3: THE REPLAY BUFFER (THE "MEMORY")
# ----------------------------------------------------------------------------
# A standard DQN component to store experiences (s, a, r, s')

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size))
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
        
    def __len__(self):
        return len(self.buffer)

# ----------------------------------------------------------------------------
# PART 4: THE TRAINING LOOP & HELPER FUNCTIONS
# ----------------------------------------------------------------------------
# This is the "main" function that orchestrates the entire
# data collection and learning process.

def select_action(state, agent, epsilon):
    """
    Selects an action using an epsilon-greedy policy.
    """
    if random.random() > epsilon:
        # Exploit: Choose the best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)
            action = q_values.argmax(dim=1).item()
    else:
        # Explore: Choose a random action
        action = random.randint(0, agent.action_dim - 1)
    return action

def train_step(agent, target_agent, buffer, batch_size, optimizer, gamma):
    """
    Performs a single step of learning from a batch of memories.
    """
    if len(buffer) < batch_size:
        return  # Not enough memories to learn yet

    # 1. Sample a batch from the buffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    # Convert numpy arrays to PyTorch tensors
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions).unsqueeze(1) # [B, 1]
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1) # [B, 1]
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.BoolTensor(dones).unsqueeze(1)    # [B, 1]

    # --- This is the Core of Q-Learning ---

    # 2. Get Predicted SQ-Values (The "Guess")
    # Get Q-values for all actions: [B, action_dim]
    all_sq_values = agent(states_t)
    # Get the specific Q-value for the action we *actually took*: [B, 1]
    predicted_sq_values = all_sq_values.gather(1, actions_t)

    # 3. Get Target SQ-Values (The "Ground Truth" via Bellman Eq.)
    with torch.no_grad():
        # Get the SQ-values for the *next* state: [B, action_dim]
        next_sq_values = target_agent(next_states_t)
        # Find the *best possible* action from the next state: [B, 1]
        max_next_sq = next_sq_values.max(dim=1, keepdim=True)[0]
        
        # *** THE CRITICAL PROSPECT THEORY STEP ***
        # We must pass the objective reward 'r' through our
        # subjective utility layer 'v(r)'
        subjective_reward = agent.utility_layer(rewards_t)
        
        # Bellman Equation: Target = v(r) + γ * max_a(SQ(s', a'))
        target_sq_values = subjective_reward + (~dones_t) * gamma * max_next_sq

    # 4. Calculate Loss (How wrong was the guess?)
    loss = F.mse_loss(predicted_sq_values, target_sq_values)

    # 5. Backpropagation (The Learning)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def plot_results(rewards):
    """
    Generates a plot of rewards per episode.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Also plot a rolling average
    if len(rewards) >= 10:
        rolling_avg = pd.Series(rewards).rolling(10).mean()
        plt.plot(rolling_avg, label='10-Episode Rolling Avg', color='orange', linewidth=2)
        plt.legend()
        
    plt.grid(True)
    plt.savefig('results.png')
    print("Plot saved to results.png")

# ----------------------------------------------------------------------------
# PART 5: EXECUTION (HOW TO RUN IT)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- 1. **THIS IS WHERE YOU FEED YOUR DATA** ---
    print("Loading QQQ market data...")
    try:
        # Load the CSV
        df = pd.read_csv('qqq_market_data.csv')
        
        # Ensure all data is numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna() # Drop any rows with missing values
        
        print(f"Data loaded. Shape: {df.shape}")
        
        # --- FEATURE ENGINEERING: USE RETURNS INSTEAD OF RAW PRICES ---
        # Raw prices are hard for NN to learn (unbounded). Returns are better (centered around 0).
        # We calculate Log Returns: ln(P_t / P_{t-1})
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna() # First row will be NaN after shift
        
        # Create a copy for the environment
        market_data = df.copy()
        
        # Normalize features
        # We will use 'log_return' and other features as inputs to the agent
        # 'price' is still kept in the dataframe for the Environment to calculate real PnL
        
        feature_cols = [c for c in market_data.columns if c != 'price']
        print(f"Using features: {feature_cols}")
        
        for col in feature_cols:
            market_data[col] = (market_data[col] - market_data[col].mean()) / (market_data[col].std() + 1e-8)
            
        print("Features normalized (including returns).")

    except FileNotFoundError:
        print("Error: qqq_market_data.csv not found. Please make sure the file is in the same directory.")
        exit()

    # --- 2. Hyperparameters ---
    NUM_EPISODES = 1000      # Increased to 1000 for better learning
    BATCH_SIZE = 32         
    GAMMA = 0.99            
    EPS_START = 1.0         
    EPS_END = 0.01          
    EPS_DECAY = 0.995       # Slow decay: 0.995^1000 ~ 0.006 (good for 1000 eps)
    TARGET_UPDATE = 10      
    WINDOW_SIZE = 20        # Increased window size to 20 days

    # --- 3. Initialization ---
    print("Initializing environment and agent...")
    env = FinancialEnv(market_data, window_size=WINDOW_SIZE)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = ProspectTheoryQNetwork(state_dim, action_dim)
    target_agent = ProspectTheoryQNetwork(state_dim, action_dim)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval() # Target network is for evaluation only
    
    optimizer = optim.Adam(agent.parameters(), lr=0.0005)
    buffer = ReplayBuffer(capacity=10000)
    
    episode_rewards = []
    epsilon = EPS_START

    # --- 4. The Main Training Loop ---
    print(f"Starting training for {NUM_EPISODES} episodes...")
    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. Agent selects an action
            action = select_action(state, agent, epsilon)
            
            # 2. Environment gives feedback
            next_state, reward, done = env.step(action)
            
            # 3. Store experience in memory
            buffer.push(state, action, reward, next_state, done)
            
            # 4. Move to next state
            state = next_state
            total_reward += reward
            
            # 5. Learn from memory
            train_step(agent, target_agent, buffer, BATCH_SIZE, optimizer, GAMMA)
            
        # --- End of Episode ---
        episode_rewards.append(total_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY) # Decay exploration
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_agent.load_state_dict(agent.state_dict())
            
        print(f"Episode: {episode+1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    print("Training complete.")
    
    # --- Save the Model ---
    torch.save(agent.state_dict(), 'prospect_theory_model.pth')
    print("Model saved to prospect_theory_model.pth")

    # --- 5. Show Results (The Output Graph) ---
    print("Plotting results...")
    plot_results(episode_rewards)
