import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
Q-VALUE SPREAD ANALYSIS
Tests if Q-value spread is a good predictor of profitable trades.
Also tests different decision frequencies (daily vs weekly).
"""

def load_model(model_path, state_dim, action_dim):
    """Loads the trained model."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def analyze_spread_vs_performance(model, data, window_size=20):
    """
    Analyzes if Q-value spread predicts profitable trades.
    
    Returns:
        DataFrame with: step, action, spread, reward, cumulative_reward
    """
    env = FinancialEnv(data, window_size=window_size)
    
    results = []
    state = env.reset()
    done = False
    step = 0
    cumulative_reward = 0
    
    with torch.no_grad():
        while not done:
            # Get Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            
            # Calculate spread (difference between best and 2nd best)
            sorted_q = np.sort(q_values)[::-1]
            spread = sorted_q[0] - sorted_q[1]
            
            # Get action
            action = np.argmax(q_values)
            
            # Take action and get reward
            next_state, reward, done = env.step(action)
            cumulative_reward += reward
            
            # Store results
            results.append({
                'step': step,
                'action': action,
                'spread': spread,
                'reward': reward,
                'cumulative_reward': cumulative_reward,
                'q_hold': q_values[0],
                'q_buy': q_values[1],
                'q_sell': q_values[2]
            })
            
            state = next_state
            step += 1
    
    return pd.DataFrame(results)

def test_decision_frequencies(df, spreads=[0, 10, 20, 50], frequencies=[1, 3, 5, 7, 10]):
    """
    Tests different decision frequencies and spread thresholds.
    
    Args:
        df: Results DataFrame from analyze_spread_vs_performance
        spreads: List of minimum spread thresholds to test
        frequencies: List of decision frequencies (in days) to test
    
    Returns:
        DataFrame with performance metrics for each combination
    """
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    
    results = []
    
    for min_spread in spreads:
        for freq in frequencies:
            # Filter by spread threshold
            if min_spread > 0:
                filtered_df = df[df['spread'] >= min_spread].copy()
            else:
                filtered_df = df.copy()
            
            # Sample every N days
            if freq > 1:
                sampled_df = filtered_df.iloc[::freq].copy()
            else:
                sampled_df = filtered_df.copy()
            
            if len(sampled_df) == 0:
                continue
            
            # Calculate metrics
            total_reward = sampled_df['reward'].sum()
            avg_reward = sampled_df['reward'].mean()
            win_rate = (sampled_df['reward'] > 0).sum() / len(sampled_df) * 100
            num_trades = len(sampled_df)
            
            # Action distribution
            buy_pct = (sampled_df['action'] == 1).sum() / len(sampled_df) * 100
            sell_pct = (sampled_df['action'] == 2).sum() / len(sampled_df) * 100
            hold_pct = (sampled_df['action'] == 0).sum() / len(sampled_df) * 100
            
            results.append({
                'min_spread': min_spread,
                'frequency_days': freq,
                'num_trades': num_trades,
                'total_reward': total_reward,
                'avg_reward_per_trade': avg_reward,
                'win_rate': win_rate,
                'buy_pct': buy_pct,
                'sell_pct': sell_pct,
                'hold_pct': hold_pct
            })
    
    return pd.DataFrame(results)

def plot_spread_analysis(df):
    """Creates visualizations of spread vs performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Spread distribution
    axes[0, 0].hist(df['spread'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=10, color='orange', linestyle='--', label='10 points')
    axes[0, 0].axvline(x=20, color='green', linestyle='--', label='20 points')
    axes[0, 0].axvline(x=50, color='red', linestyle='--', label='50 points')
    axes[0, 0].set_title('Q-Value Spread Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Spread (points)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spread vs Reward (scatter)
    axes[0, 1].scatter(df['spread'], df['reward'], alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Spread vs Immediate Reward', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Spread (points)')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average reward by spread bucket
    spread_buckets = pd.cut(df['spread'], bins=[0, 10, 20, 50, 100, df['spread'].max()])
    avg_rewards = df.groupby(spread_buckets)['reward'].mean()
    win_rates = df.groupby(spread_buckets).apply(lambda x: (x['reward'] > 0).sum() / len(x) * 100)
    
    x_labels = ['0-10', '10-20', '20-50', '50-100', f'100+']
    x_pos = np.arange(len(avg_rewards))
    
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    bars = ax3.bar(x_pos, avg_rewards.values, alpha=0.7, color='blue', label='Avg Reward')
    line = ax3_twin.plot(x_pos, win_rates.values, 'ro-', linewidth=2, markersize=8, label='Win Rate')
    
    ax3.set_xlabel('Spread Range (points)')
    ax3.set_ylabel('Average Reward', color='blue')
    ax3_twin.set_ylabel('Win Rate (%)', color='red')
    ax3.set_title('Performance by Spread Range', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 4: Cumulative reward over time
    axes[1, 1].plot(df['step'], df['cumulative_reward'], linewidth=1.5)
    axes[1, 1].set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].fill_between(df['step'], df['cumulative_reward'], alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spread_analysis.png', dpi=150)
    print("\nSpread analysis chart saved to spread_analysis.png")

if __name__ == "__main__":
    print("=" * 70)
    print("📊 Q-VALUE SPREAD ANALYSIS")
    print("=" * 70)
    
    # Configuration
    data_file = 'qqq_market_data.csv'
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    # Load data
    print(f"\n📊 Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna()
        
        feature_cols = [c for c in df.columns if c != 'price']
        for col in feature_cols:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        print(f"✅ Data loaded: {len(df)} days")
    except FileNotFoundError:
        print(f"❌ Error: {data_file} not found.")
        exit()
    
    # Load model
    print(f"\n🧠 Loading model from {model_path}...")
    try:
        temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
        agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
        print(f"✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: {model_path} not found.")
        exit()
    
    # Analyze spread vs performance
    print(f"\n🔍 Analyzing Q-value spreads and performance...")
    results_df = analyze_spread_vs_performance(agent, df, WINDOW_SIZE)
    
    # Spread statistics
    print("\n" + "=" * 70)
    print("📊 SPREAD STATISTICS")
    print("=" * 70)
    
    print(f"\n📈 Overall Spread Statistics:")
    print(f"  Average Spread: {results_df['spread'].mean():.2f} points")
    print(f"  Median Spread: {results_df['spread'].median():.2f} points")
    print(f"  Max Spread: {results_df['spread'].max():.2f} points")
    print(f"  Min Spread: {results_df['spread'].min():.2f} points")
    print(f"  Std Deviation: {results_df['spread'].std():.2f} points")
    
    # Spread distribution
    very_low = len(results_df[results_df['spread'] < 10])
    low = len(results_df[(results_df['spread'] >= 10) & (results_df['spread'] < 20)])
    medium = len(results_df[(results_df['spread'] >= 20) & (results_df['spread'] < 50)])
    high = len(results_df[results_df['spread'] >= 50])
    
    total = len(results_df)
    
    print(f"\n📊 Spread Distribution:")
    print(f"  <10 points:    {very_low:4d} days ({very_low/total*100:5.1f}%)")
    print(f"  10-20 points:  {low:4d} days ({low/total*100:5.1f}%)")
    print(f"  20-50 points:  {medium:4d} days ({medium/total*100:5.1f}%)")
    print(f"  ≥50 points:    {high:4d} days ({high/total*100:5.1f}%)")
    
    # Performance by spread range
    print(f"\n💰 Performance by Spread Range:")
    
    for min_spread, max_spread, label in [(0, 10, '<10'), (10, 20, '10-20'), (20, 50, '20-50'), (50, 1000, '≥50')]:
        subset = results_df[(results_df['spread'] >= min_spread) & (results_df['spread'] < max_spread)]
        if len(subset) > 0:
            avg_reward = subset['reward'].mean()
            win_rate = (subset['reward'] > 0).sum() / len(subset) * 100
            total_reward = subset['reward'].sum()
            print(f"  {label:6s} points: {len(subset):4d} trades | Avg: {avg_reward:6.3f} | Win Rate: {win_rate:5.1f}% | Total: {total_reward:7.1f}")
    
    # Test different strategies
    print(f"\n🧪 Testing Different Strategies...")
    strategy_results = test_decision_frequencies(
        results_df,
        spreads=[0, 10, 20, 50],
        frequencies=[1, 3, 5, 7, 10, 15, 20]
    )
    
    # Display top strategies
    print("\n" + "=" * 70)
    print("🏆 TOP STRATEGIES (by Total Reward)")
    print("=" * 70)
    
    top_strategies = strategy_results.nlargest(10, 'total_reward')
    print(f"\n{'Spread':>8s} | {'Freq':>5s} | {'Trades':>7s} | {'Total':>8s} | {'Avg/Trade':>10s} | {'Win%':>6s}")
    print("-" * 70)
    for _, row in top_strategies.iterrows():
        print(f"{row['min_spread']:>7.0f}+ | {row['frequency_days']:>4.0f}d | {row['num_trades']:>7.0f} | {row['total_reward']:>8.1f} | {row['avg_reward_per_trade']:>10.3f} | {row['win_rate']:>5.1f}%")
    
    print("\n" + "=" * 70)
    print("🏆 TOP STRATEGIES (by Average Reward per Trade)")
    print("=" * 70)
    
    top_avg = strategy_results.nlargest(10, 'avg_reward_per_trade')
    print(f"\n{'Spread':>8s} | {'Freq':>5s} | {'Trades':>7s} | {'Total':>8s} | {'Avg/Trade':>10s} | {'Win%':>6s}")
    print("-" * 70)
    for _, row in top_avg.iterrows():
        print(f"{row['min_spread']:>7.0f}+ | {row['frequency_days']:>4.0f}d | {row['num_trades']:>7.0f} | {row['total_reward']:>8.1f} | {row['avg_reward_per_trade']:>10.3f} | {row['win_rate']:>5.1f}%")
    
    # Key insights
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)
    
    # Compare daily vs weekly
    daily_all = strategy_results[(strategy_results['min_spread'] == 0) & (strategy_results['frequency_days'] == 1)]
    weekly_all = strategy_results[(strategy_results['min_spread'] == 0) & (strategy_results['frequency_days'] == 7)]
    
    if len(daily_all) > 0 and len(weekly_all) > 0:
        daily_reward = daily_all.iloc[0]['total_reward']
        weekly_reward = weekly_all.iloc[0]['total_reward']
        daily_trades = daily_all.iloc[0]['num_trades']
        weekly_trades = weekly_all.iloc[0]['num_trades']
        
        print(f"\n📊 Daily vs Weekly Trading:")
        print(f"  Daily (all signals):   {daily_trades:.0f} trades, {daily_reward:.1f} total reward")
        print(f"  Weekly (all signals):  {weekly_trades:.0f} trades, {weekly_reward:.1f} total reward")
        print(f"  Difference: {weekly_reward - daily_reward:+.1f} ({(weekly_reward/daily_reward - 1)*100:+.1f}%)")
    
    # Best spread threshold
    best_spread = strategy_results[strategy_results['frequency_days'] == 1].nlargest(1, 'total_reward')
    if len(best_spread) > 0:
        print(f"\n🎯 Best Spread Threshold (daily trading):")
        print(f"  Min Spread: {best_spread.iloc[0]['min_spread']:.0f}+ points")
        print(f"  Trades: {best_spread.iloc[0]['num_trades']:.0f}")
        print(f"  Total Reward: {best_spread.iloc[0]['total_reward']:.1f}")
        print(f"  Avg per Trade: {best_spread.iloc[0]['avg_reward_per_trade']:.3f}")
    
    # Create visualizations
    print(f"\n📊 Generating visualizations...")
    plot_spread_analysis(results_df)
    
    # Save results
    strategy_results.to_csv('strategy_comparison.csv', index=False)
    print(f"\n💾 Strategy comparison saved to strategy_comparison.csv")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
