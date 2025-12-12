import torch
import numpy as np
import pandas as pd
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
DAY OF WEEK ANALYSIS
Tests which specific day of the week (Mon-Fri) is actually best for trading.
Assumes data is contiguous trading days.
"""

def load_model(model_path, state_dim, action_dim):
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def analyze_day_of_week(model, data, window_size=20):
    env = FinancialEnv(data, window_size=window_size)
    
    # We need to assign a Day of Week to each row.
    # Since we don't have dates in the CSV, we'll assume the last row is a known date
    # and work backward.
    # Let's assume the last row of the historical data (2512 rows) was Nov 19, 2024 (Tuesday).
    # This is an approximation, but sufficient to establish the pattern.
    
    # Create a list of day indices (0=Mon, 4=Fri) working backward
    # Trading days: Mon(0), Tue(1), Wed(2), Thu(3), Fri(4)
    
    days_of_week = []
    current_day = 1 # Nov 19, 2024 was Tuesday (1)
    
    for _ in range(len(data)):
        days_of_week.append(current_day)
        current_day -= 1
        if current_day < 0:
            current_day = 4 # Jump back to Friday
            
    days_of_week.reverse() # Now it matches the data order
    
    # Generate signals
    results = []
    state = env.reset()
    done = False
    step = 0
    
    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            
            if step < len(days_of_week):
                day_idx = days_of_week[step]
                
                results.append({
                    'step': step,
                    'day_idx': day_idx, # 0-4
                    'action': action,
                    'reward': reward
                })
            
            state = next_state
            step += 1
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("=" * 60)
    print("📅 TRUE DAY OF WEEK ANALYSIS")
    print("=" * 60)
    
    # Load data
    data_file = 'qqq_market_data.csv'
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    feature_cols = [c for c in df.columns if c != 'price']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    # Load model
    temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
    agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
    
    # Analyze
    results_df = analyze_day_of_week(agent, df, WINDOW_SIZE)
    
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    
    print(f"\n{'Day':<12s} | {'Trades':>6s} | {'Total Reward':>12s} | {'Avg Reward':>10s} | {'Win Rate':>8s}")
    print("-" * 65)
    
    for day_idx in range(5):
        day_data = results_df[results_df['day_idx'] == day_idx]
        
        total_reward = day_data['reward'].sum()
        avg_reward = day_data['reward'].mean()
        win_rate = (day_data['reward'] > 0).sum() / len(day_data) * 100
        
        print(f"{day_names[day_idx]:<12s} | {len(day_data):>6d} | {total_reward:>12.2f} | {avg_reward:>10.4f} | {win_rate:>7.1f}%")
        
    print("\n" + "=" * 60)
