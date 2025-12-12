import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
CONFIDENCE ANALYSIS SCRIPT
Analyzes the model's confidence levels across all historical data.
"""

def load_model(model_path, state_dim, action_dim):
    """Loads the trained model."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def calculate_confidence(q_values):
    """
    Calculates confidence as percentage difference between best and 2nd best action.
    
    Args:
        q_values: Array of Q-values [Hold, Buy, Sell]
    
    Returns:
        confidence_pct: Confidence percentage
        best_action: Index of best action (0=Hold, 1=Buy, 2=Sell)
    """
    best_action = np.argmax(q_values)
    sorted_q = np.sort(q_values)[::-1]
    confidence_pct = ((sorted_q[0] - sorted_q[1]) / abs(sorted_q[0]) * 100) if sorted_q[0] != 0 else 0
    return confidence_pct, best_action

def analyze_all_confidence_levels(model, data, window_size=20):
    """
    Analyzes confidence levels for every day in the dataset.
    
    Returns:
        DataFrame with columns: step, action, confidence, q_hold, q_buy, q_sell
    """
    env = FinancialEnv(data, window_size=window_size)
    
    results = []
    state = env.reset()
    done = False
    step = 0
    
    with torch.no_grad():
        while not done:
            # Get Q-values for current state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            
            # Calculate confidence
            confidence, action = calculate_confidence(q_values)
            
            # Store results
            results.append({
                'step': step,
                'action': action,
                'confidence': confidence,
                'q_hold': q_values[0],
                'q_buy': q_values[1],
                'q_sell': q_values[2]
            })
            
            # Take action and move to next state
            next_state, _, done = env.step(action)
            state = next_state
            step += 1
    
    return pd.DataFrame(results)

def plot_confidence_distribution(df):
    """Creates visualizations of confidence patterns."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Confidence over time
    axes[0, 0].plot(df['step'], df['confidence'], alpha=0.7, linewidth=0.8)
    axes[0, 0].axhline(y=10, color='orange', linestyle='--', label='10% threshold')
    axes[0, 0].axhline(y=15, color='green', linestyle='--', label='15% threshold')
    axes[0, 0].axhline(y=20, color='red', linestyle='--', label='20% threshold')
    axes[0, 0].set_title('Confidence Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Confidence (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence histogram
    axes[0, 1].hist(df['confidence'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=10, color='orange', linestyle='--', label='10%')
    axes[0, 1].axvline(x=15, color='green', linestyle='--', label='15%')
    axes[0, 1].axvline(x=20, color='red', linestyle='--', label='20%')
    axes[0, 1].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Confidence (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Action distribution by confidence level
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    
    # Categorize by confidence
    low_conf = df[df['confidence'] < 10]
    med_conf = df[(df['confidence'] >= 10) & (df['confidence'] < 20)]
    high_conf = df[df['confidence'] >= 20]
    
    categories = ['Low (<10%)', 'Medium (10-20%)', 'High (>20%)']
    hold_counts = [
        len(low_conf[low_conf['action'] == 0]),
        len(med_conf[med_conf['action'] == 0]),
        len(high_conf[high_conf['action'] == 0])
    ]
    buy_counts = [
        len(low_conf[low_conf['action'] == 1]),
        len(med_conf[med_conf['action'] == 1]),
        len(high_conf[high_conf['action'] == 1])
    ]
    sell_counts = [
        len(low_conf[low_conf['action'] == 2]),
        len(med_conf[med_conf['action'] == 2]),
        len(high_conf[high_conf['action'] == 2])
    ]
    
    x = np.arange(len(categories))
    width = 0.25
    
    axes[1, 0].bar(x - width, hold_counts, width, label='Hold', color='gray')
    axes[1, 0].bar(x, buy_counts, width, label='Buy', color='green')
    axes[1, 0].bar(x + width, sell_counts, width, label='Sell', color='red')
    axes[1, 0].set_title('Actions by Confidence Level', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Confidence Level')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative confidence distribution
    sorted_conf = np.sort(df['confidence'])
    cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf) * 100
    
    axes[1, 1].plot(sorted_conf, cumulative, linewidth=2)
    axes[1, 1].axvline(x=10, color='orange', linestyle='--', label='10%')
    axes[1, 1].axvline(x=15, color='green', linestyle='--', label='15%')
    axes[1, 1].axvline(x=20, color='red', linestyle='--', label='20%')
    axes[1, 1].set_title('Cumulative Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Confidence (%)')
    axes[1, 1].set_ylabel('Cumulative % of Days')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=150)
    print("\nConfidence analysis chart saved to confidence_analysis.png")

if __name__ == "__main__":
    print("=" * 70)
    print("📊 CONFIDENCE LEVEL ANALYSIS")
    print("=" * 70)
    
    # Configuration
    data_file = 'qqq_market_data.csv'  # Use full historical data
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    # Load data
    print(f"\n📊 Loading historical data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        # Apply same preprocessing as training
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna()
        
        # Normalize features
        feature_cols = [c for c in df.columns if c != 'price']
        for col in feature_cols:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        print(f"✅ Data loaded: {len(df)} days of data")
        
    except FileNotFoundError:
        print(f"❌ Error: {data_file} not found.")
        exit()
    
    # Load model
    print(f"\n🧠 Loading trained model from {model_path}...")
    try:
        temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
        agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
        print(f"✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: {model_path} not found. Train the model first.")
        exit()
    
    # Analyze confidence levels
    print(f"\n🔍 Analyzing confidence levels across all {len(df)} days...")
    results_df = analyze_all_confidence_levels(agent, df, WINDOW_SIZE)
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("📈 CONFIDENCE STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Days Analyzed: {len(results_df)}")
    print(f"  Average Confidence: {results_df['confidence'].mean():.2f}%")
    print(f"  Median Confidence: {results_df['confidence'].median():.2f}%")
    print(f"  Max Confidence: {results_df['confidence'].max():.2f}%")
    print(f"  Min Confidence: {results_df['confidence'].min():.2f}%")
    print(f"  Std Deviation: {results_df['confidence'].std():.2f}%")
    
    # Confidence level breakdown
    very_low = len(results_df[results_df['confidence'] < 5])
    low = len(results_df[(results_df['confidence'] >= 5) & (results_df['confidence'] < 10)])
    medium = len(results_df[(results_df['confidence'] >= 10) & (results_df['confidence'] < 15)])
    high = len(results_df[(results_df['confidence'] >= 15) & (results_df['confidence'] < 20)])
    very_high = len(results_df[results_df['confidence'] >= 20])
    
    total = len(results_df)
    
    print(f"\n📊 Confidence Level Breakdown:")
    print(f"  Very Low (<5%):    {very_low:4d} days ({very_low/total*100:5.1f}%)")
    print(f"  Low (5-10%):       {low:4d} days ({low/total*100:5.1f}%)")
    print(f"  Medium (10-15%):   {medium:4d} days ({medium/total*100:5.1f}%)")
    print(f"  High (15-20%):     {high:4d} days ({high/total*100:5.1f}%)")
    print(f"  Very High (>20%):  {very_high:4d} days ({very_high/total*100:5.1f}%)")
    
    # Key thresholds
    above_10 = len(results_df[results_df['confidence'] >= 10])
    above_15 = len(results_df[results_df['confidence'] >= 15])
    above_20 = len(results_df[results_df['confidence'] >= 20])
    
    print(f"\n🎯 Key Thresholds:")
    print(f"  Confidence ≥10%: {above_10:4d} days ({above_10/total*100:5.1f}%)")
    print(f"  Confidence ≥15%: {above_15:4d} days ({above_15/total*100:5.1f}%)")
    print(f"  Confidence ≥20%: {above_20:4d} days ({above_20/total*100:5.1f}%)")
    
    # Action distribution
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    print(f"\n📊 Action Distribution:")
    for action_id, action_name in action_names.items():
        count = len(results_df[results_df['action'] == action_id])
        print(f"  {action_name}: {count:4d} days ({count/total*100:5.1f}%)")
    
    # High confidence actions
    high_conf_df = results_df[results_df['confidence'] >= 15]
    if len(high_conf_df) > 0:
        print(f"\n🎯 High Confidence (≥15%) Action Distribution:")
        for action_id, action_name in action_names.items():
            count = len(high_conf_df[high_conf_df['action'] == action_id])
            print(f"  {action_name}: {count:4d} days ({count/len(high_conf_df)*100:5.1f}%)")
    
    # Top 10 most confident days
    print(f"\n🏆 Top 10 Most Confident Days:")
    top_10 = results_df.nlargest(10, 'confidence')
    for idx, row in top_10.iterrows():
        action_name = action_names[row['action']]
        print(f"  Day {int(row['step']):4d}: {action_name:4s} - {row['confidence']:6.2f}% confidence")
    
    # Create visualizations
    print(f"\n📊 Generating confidence analysis charts...")
    plot_confidence_distribution(results_df)
    
    # Save detailed results
    results_df.to_csv('confidence_history.csv', index=False)
    print(f"\n💾 Detailed results saved to confidence_history.csv")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
