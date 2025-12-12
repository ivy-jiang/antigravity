import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
INFERENCE SCRIPT
This script loads a pre-trained model and runs it on new data WITHOUT training.
Use this to test your model on future data or to make predictions.
"""

def load_model(model_path, state_dim, action_dim):
    """
    Loads a saved model from disk.
    
    Args:
        model_path: Path to the .pth file
        state_dim: State dimension (must match training)
        action_dim: Action dimension (must match training)
    
    Returns:
        Loaded model ready for inference
    """
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    print(f"Model loaded from {model_path}")
    return model

def run_inference(model, env):
    """
    Runs the trained model on an environment without training.
    
    Args:
        model: The loaded model
        env: The environment with new data
    
    Returns:
        total_reward: The total reward earned
        actions_taken: List of actions taken at each step
    """
    state = env.reset()
    total_reward = 0
    done = False
    actions_taken = []
    
    with torch.no_grad():  # No gradient calculation needed for inference
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get Q-values from the model
            q_values = model(state_tensor)
            
            # Choose the BEST action (no exploration, epsilon = 0)
            action = q_values.argmax(dim=1).item()
            actions_taken.append(action)
            
            # Take the action
            next_state, reward, done = env.step(action)
            
            state = next_state
            total_reward += reward
    
    return total_reward, actions_taken

def analyze_actions(actions_taken):
    """
    Analyzes the distribution of actions taken.
    
    Args:
        actions_taken: List of actions (0=Hold, 1=Buy, 2=Sell)
    
    Returns:
        Dictionary with action counts
    """
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    action_counts = {name: 0 for name in action_names.values()}
    
    for action in actions_taken:
        action_counts[action_names[action]] += 1
    
    return action_counts

if __name__ == "__main__":
    print("=" * 60)
    print("INFERENCE MODE - Testing Pre-Trained Model")
    print("=" * 60)
    
    # --- 1. Load Your Test Data ---
    # Replace this with your actual test data file
    test_data_file = 'qqq_market_data.csv'  # Change this to your test file
    
    print(f"\nLoading test data from {test_data_file}...")
    try:
        df = pd.read_csv(test_data_file)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        # Apply the SAME feature engineering as training
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna()
        
        test_data = df.copy()
        
        # Normalize features (same as training)
        feature_cols = [c for c in test_data.columns if c != 'price']
        for col in feature_cols:
            test_data[col] = (test_data[col] - test_data[col].mean()) / (test_data[col].std() + 1e-8)
        
        print(f"Test data loaded. Shape: {test_data.shape}")
        
    except FileNotFoundError:
        print(f"Error: {test_data_file} not found.")
        exit()
    
    # --- 2. Initialize Environment ---
    WINDOW_SIZE = 20  # MUST match training
    env = FinancialEnv(test_data, window_size=WINDOW_SIZE)
    
    # --- 3. Load the Trained Model ---
    model_path = 'prospect_theory_model.pth'
    try:
        agent = load_model(model_path, env.state_dim, env.action_dim)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        exit()
    
    # --- 4. Run Inference ---
    print("\nRunning inference on test data...")
    total_reward, actions_taken = run_inference(agent, env)
    
    # --- 5. Display Results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Number of Steps: {len(actions_taken)}")
    
    action_distribution = analyze_actions(actions_taken)
    print("\nAction Distribution:")
    for action_name, count in action_distribution.items():
        percentage = (count / len(actions_taken)) * 100
        print(f"  {action_name}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)
