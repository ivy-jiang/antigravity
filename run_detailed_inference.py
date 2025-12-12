import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
DETAILED INFERENCE SCRIPT
This script loads a pre-trained model and analyzes day-to-day performance.
"""

def load_model(model_path, state_dim, action_dim):
    """Loads a saved model from disk."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def run_detailed_inference(model, env):
    """
    Runs the trained model and tracks detailed step-by-step performance.
    
    Returns:
        step_rewards: List of rewards at each step
        actions_taken: List of actions at each step
        cumulative_rewards: Running total of rewards
    """
    state = env.reset()
    step_rewards = []
    actions_taken = []
    cumulative_rewards = []
    cumulative_total = 0
    
    done = False
    step = 0
    
    with torch.no_grad():
        while not done:
            # Get action from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Track performance
            step_rewards.append(reward)
            actions_taken.append(action)
            cumulative_total += reward
            cumulative_rewards.append(cumulative_total)
            
            state = next_state
            step += 1
    
    return step_rewards, actions_taken, cumulative_rewards

def analyze_consistency(step_rewards):
    """
    Analyzes the consistency of day-to-day performance.
    
    Returns:
        Dictionary with consistency metrics
    """
    step_rewards = np.array(step_rewards)
    
    # Calculate metrics
    positive_days = np.sum(step_rewards > 0)
    negative_days = np.sum(step_rewards < 0)
    neutral_days = np.sum(step_rewards == 0)
    total_days = len(step_rewards)
    
    # Win rate
    win_rate = (positive_days / total_days) * 100
    
    # Average profit/loss
    avg_profit = np.mean(step_rewards[step_rewards > 0]) if positive_days > 0 else 0
    avg_loss = np.mean(step_rewards[step_rewards < 0]) if negative_days > 0 else 0
    
    # Volatility (standard deviation)
    volatility = np.std(step_rewards)
    
    # Max drawdown (worst consecutive loss)
    cumulative = np.cumsum(step_rewards)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown)
    
    # Longest winning/losing streaks
    def get_streaks(rewards, target):
        streaks = []
        current_streak = 0
        for r in rewards:
            if (target == 'win' and r > 0) or (target == 'loss' and r < 0):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        return max(streaks) if streaks else 0
    
    longest_win_streak = get_streaks(step_rewards, 'win')
    longest_loss_streak = get_streaks(step_rewards, 'loss')
    
    return {
        'total_days': total_days,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'neutral_days': neutral_days,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'longest_win_streak': longest_win_streak,
        'longest_loss_streak': longest_loss_streak
    }

def plot_performance(step_rewards, cumulative_rewards, actions_taken):
    """Creates visualizations of the model's performance."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Daily Rewards
    axes[0].plot(step_rewards, alpha=0.7, linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Daily Rewards (Step-by-Step)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Add rolling average
    window = 50
    if len(step_rewards) >= window:
        rolling_avg = pd.Series(step_rewards).rolling(window).mean()
        axes[0].plot(rolling_avg, color='orange', linewidth=2, label=f'{window}-Day Moving Avg')
        axes[0].legend()
    
    # Plot 2: Cumulative Rewards
    axes[1].plot(cumulative_rewards, color='green', linewidth=1.5)
    axes[1].set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Cumulative Reward')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(range(len(cumulative_rewards)), cumulative_rewards, alpha=0.3, color='green')
    
    # Plot 3: Action Distribution Over Time
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    colors = {0: 'gray', 1: 'green', 2: 'red'}
    
    for action_id, action_name in action_names.items():
        action_mask = [1 if a == action_id else 0 for a in actions_taken]
        # Use rolling sum to show action frequency over time
        if len(action_mask) >= window:
            rolling_action = pd.Series(action_mask).rolling(window).sum()
            axes[2].plot(rolling_action, label=action_name, color=colors[action_id], linewidth=1.5)
    
    axes[2].set_title(f'Action Frequency ({window}-Day Rolling Window)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Day')
    axes[2].set_ylabel(f'Count (per {window} days)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_performance.png', dpi=150)
    print("\nDetailed performance chart saved to detailed_performance.png")

if __name__ == "__main__":
    print("=" * 70)
    print("DETAILED INFERENCE - Day-to-Day Performance Analysis")
    print("=" * 70)
    
    # Load data
    test_data_file = 'qqq_market_data.csv'
    print(f"\nLoading test data from {test_data_file}...")
    
    try:
        df = pd.read_csv(test_data_file)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        # Apply same feature engineering as training
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna()
        
        test_data = df.copy()
        
        # Normalize features
        feature_cols = [c for c in test_data.columns if c != 'price']
        for col in feature_cols:
            test_data[col] = (test_data[col] - test_data[col].mean()) / (test_data[col].std() + 1e-8)
        
        print(f"Test data loaded. Shape: {test_data.shape}")
        
    except FileNotFoundError:
        print(f"Error: {test_data_file} not found.")
        exit()
    
    # Initialize environment
    WINDOW_SIZE = 20
    env = FinancialEnv(test_data, window_size=WINDOW_SIZE)
    
    # Load model
    model_path = 'prospect_theory_model.pth'
    try:
        agent = load_model(model_path, env.state_dim, env.action_dim)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        exit()
    
    # Run detailed inference
    print("\nRunning detailed inference...")
    step_rewards, actions_taken, cumulative_rewards = run_detailed_inference(agent, env)
    
    # Analyze consistency
    metrics = analyze_consistency(step_rewards)
    
    # Display results
    print("\n" + "=" * 70)
    print("DAY-TO-DAY CONSISTENCY ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total Days Traded: {metrics['total_days']}")
    print(f"  Total Reward: {sum(step_rewards):.2f}")
    print(f"  Average Daily Reward: {np.mean(step_rewards):.4f}")
    
    print(f"\n✅ Win/Loss Statistics:")
    print(f"  Profitable Days: {metrics['positive_days']} ({metrics['win_rate']:.1f}%)")
    print(f"  Losing Days: {metrics['negative_days']} ({(metrics['negative_days']/metrics['total_days'])*100:.1f}%)")
    print(f"  Neutral Days: {metrics['neutral_days']} ({(metrics['neutral_days']/metrics['total_days'])*100:.1f}%)")
    
    print(f"\n💰 Profit/Loss Analysis:")
    print(f"  Average Profit (on winning days): {metrics['avg_profit']:.4f}")
    print(f"  Average Loss (on losing days): {metrics['avg_loss']:.4f}")
    print(f"  Profit/Loss Ratio: {abs(metrics['avg_profit']/metrics['avg_loss']):.2f}x" if metrics['avg_loss'] != 0 else "  Profit/Loss Ratio: N/A")
    
    print(f"\n📈 Consistency Metrics:")
    print(f"  Daily Volatility (Std Dev): {metrics['volatility']:.4f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}")
    print(f"  Longest Winning Streak: {metrics['longest_win_streak']} days")
    print(f"  Longest Losing Streak: {metrics['longest_loss_streak']} days")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if metrics['win_rate'] > 60:
        print(f"  ✅ EXCELLENT: Win rate of {metrics['win_rate']:.1f}% shows strong consistency")
    elif metrics['win_rate'] > 50:
        print(f"  ✅ GOOD: Win rate of {metrics['win_rate']:.1f}% is above 50%")
    else:
        print(f"  ⚠️  CAUTION: Win rate of {metrics['win_rate']:.1f}% is below 50%")
    
    if metrics['volatility'] < 1.0:
        print(f"  ✅ LOW VOLATILITY: Daily returns are relatively stable")
    elif metrics['volatility'] < 2.0:
        print(f"  ⚠️  MODERATE VOLATILITY: Some day-to-day variation")
    else:
        print(f"  ⚠️  HIGH VOLATILITY: Significant day-to-day swings")
    
    # Create visualizations
    print("\nGenerating performance charts...")
    plot_performance(step_rewards, cumulative_rewards, actions_taken)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
