import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
ADVANCED TRADING STRATEGY ANALYSIS
Tests position holding, PnL calculation, and weekly trading variations.
"""

def load_model(model_path, state_dim, action_dim):
    """Loads the trained model."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_all_signals(model, data, window_size=20):
    """
    Gets all trading signals with Q-values and spreads.
    """
    env = FinancialEnv(data, window_size=window_size)
    
    results = []
    state = env.reset()
    done = False
    step = 0
    
    # Get actual prices for PnL calculation
    prices = data['price'].values
    
    with torch.no_grad():
        while not done:
            # Get Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            
            # Calculate spread
            sorted_q = np.sort(q_values)[::-1]
            spread = sorted_q[0] - sorted_q[1]
            
            # Get action
            action = np.argmax(q_values)
            
            # Take action and get reward
            next_state, reward, done = env.step(action)
            
            # Store results
            results.append({
                'step': step,
                'action': action,
                'spread': spread,
                'reward': reward,
                'price': prices[step + window_size] if step + window_size < len(prices) else prices[-1],
                'q_hold': q_values[0],
                'q_buy': q_values[1],
                'q_sell': q_values[2]
            })
            
            state = next_state
            step += 1
    
    return pd.DataFrame(results)

def simulate_threshold_strategy(signals_df, min_spread=20, hold_position=True):
    """
    Simulates a strategy that only trades when spread >= min_spread.
    If hold_position=True, maintains last position when spread is low.
    If hold_position=False, goes to cash (no position) when spread is low.
    
    Returns:
        DataFrame with: step, signal_action, actual_action, position, pnl, cumulative_pnl
    """
    results = []
    current_position = 0  # 0=cash, 1=long, -1=short
    entry_price = 0
    cumulative_pnl = 0
    
    for idx, row in signals_df.iterrows():
        step = row['step']
        signal_action = row['action']
        spread = row['spread']
        price = row['price']
        
        # Determine if we should act on this signal
        if spread >= min_spread:
            # High confidence signal - take action
            actual_action = signal_action
        else:
            # Low confidence signal
            if hold_position:
                # Hold current position
                actual_action = 0  # Hold
            else:
                # Go to cash
                actual_action = 0  # Hold (which will close position below)
        
        # Calculate PnL based on position changes
        pnl = 0
        
        # Map action to position: 0=cash, 1=buy/long, 2=sell/short
        if actual_action == 1:  # Buy signal
            if current_position == 0:
                # Enter long
                current_position = 1
                entry_price = price
            elif current_position == -1:
                # Close short, enter long
                pnl = (entry_price - price) / entry_price  # Profit from short
                current_position = 1
                entry_price = price
            # If already long, do nothing
        
        elif actual_action == 2:  # Sell signal
            if current_position == 0:
                # Enter short
                current_position = -1
                entry_price = price
            elif current_position == 1:
                # Close long, enter short
                pnl = (price - entry_price) / entry_price  # Profit from long
                current_position = -1
                entry_price = price
            # If already short, do nothing
        
        elif actual_action == 0:  # Hold signal
            if not hold_position and current_position != 0:
                # Close position and go to cash
                if current_position == 1:
                    pnl = (price - entry_price) / entry_price
                elif current_position == -1:
                    pnl = (entry_price - price) / entry_price
                current_position = 0
                entry_price = 0
        
        cumulative_pnl += pnl
        
        results.append({
            'step': step,
            'signal_action': signal_action,
            'actual_action': actual_action,
            'spread': spread,
            'position': current_position,
            'price': price,
            'pnl': pnl,
            'cumulative_pnl': cumulative_pnl
        })
    
    return pd.DataFrame(results)

def simulate_weekly_strategy(signals_df, min_spread=0, day_of_week=0):
    """
    Simulates weekly trading strategy.
    
    Args:
        signals_df: Daily signals
        min_spread: Minimum spread threshold
        day_of_week: Which day to check (0=Monday, 6=Sunday, in terms of offset)
    
    Returns:
        DataFrame with weekly results
    """
    results = []
    current_position = 0
    entry_price = 0
    cumulative_pnl = 0
    
    # Sample every 7 days starting from day_of_week
    weekly_signals = signals_df.iloc[day_of_week::7].copy()
    
    for idx, row in weekly_signals.iterrows():
        step = row['step']
        signal_action = row['action']
        spread = row['spread']
        price = row['price']
        
        # Only act if spread meets threshold
        if spread < min_spread:
            continue
        
        actual_action = signal_action
        pnl = 0
        
        # Execute trade
        if actual_action == 1:  # Buy
            if current_position == 0:
                current_position = 1
                entry_price = price
            elif current_position == -1:
                pnl = (entry_price - price) / entry_price
                current_position = 1
                entry_price = price
        
        elif actual_action == 2:  # Sell
            if current_position == 0:
                current_position = -1
                entry_price = price
            elif current_position == 1:
                pnl = (price - entry_price) / entry_price
                current_position = -1
                entry_price = price
        
        cumulative_pnl += pnl
        
        results.append({
            'step': step,
            'action': actual_action,
            'spread': spread,
            'position': current_position,
            'price': price,
            'pnl': pnl,
            'cumulative_pnl': cumulative_pnl
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("=" * 80)
    print("📊 ADVANCED TRADING STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Configuration
    data_file = 'qqq_market_data.csv'
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    # Load data
    print(f"\n📊 Loading data...")
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    feature_cols = [c for c in df.columns if c != 'price']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    print(f"✅ Data loaded: {len(df)} days")
    
    # Load model
    print(f"\n🧠 Loading model...")
    temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
    agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
    print(f"✅ Model loaded")
    
    # Get all signals
    print(f"\n🔍 Generating all trading signals...")
    signals_df = get_all_signals(agent, df, WINDOW_SIZE)
    print(f"✅ Generated {len(signals_df)} signals")
    
    # Test different threshold strategies
    print("\n" + "=" * 80)
    print("🎯 THRESHOLD STRATEGIES (with Position Holding)")
    print("=" * 80)
    
    strategies = [
        (0, False, "Daily - All Signals (No Hold)"),
        (0, True, "Daily - All Signals (Hold Position)"),
        (20, False, "Threshold 20+ (Close on Low Confidence)"),
        (20, True, "Threshold 20+ (Hold on Low Confidence)"),
        (50, False, "Threshold 50+ (Close on Low Confidence)"),
        (50, True, "Threshold 50+ (Hold on Low Confidence)"),
    ]
    
    strategy_results = []
    
    for min_spread, hold_pos, name in strategies:
        result_df = simulate_threshold_strategy(signals_df, min_spread, hold_pos)
        
        total_pnl = result_df['cumulative_pnl'].iloc[-1] if len(result_df) > 0 else 0
        num_trades = len(result_df[result_df['pnl'] != 0])
        avg_pnl = result_df[result_df['pnl'] != 0]['pnl'].mean() if num_trades > 0 else 0
        win_rate = (result_df[result_df['pnl'] > 0].shape[0] / num_trades * 100) if num_trades > 0 else 0
        
        strategy_results.append({
            'strategy': name,
            'min_spread': min_spread,
            'hold_position': hold_pos,
            'total_pnl': total_pnl * 100,  # Convert to percentage
            'num_trades': num_trades,
            'avg_pnl': avg_pnl * 100,
            'win_rate': win_rate
        })
    
    strategy_df = pd.DataFrame(strategy_results)
    
    print(f"\n{'Strategy':<45s} | {'Trades':>7s} | {'Total PnL':>10s} | {'Avg PnL':>9s} | {'Win Rate':>8s}")
    print("-" * 80)
    for _, row in strategy_df.iterrows():
        print(f"{row['strategy']:<45s} | {row['num_trades']:>7.0f} | {row['total_pnl']:>9.2f}% | {row['avg_pnl']:>8.3f}% | {row['win_rate']:>7.1f}%")
    
    # Weekly trading analysis
    print("\n" + "=" * 80)
    print("📅 WEEKLY TRADING ANALYSIS")
    print("=" * 80)
    
    print("\n🔍 Testing different days of the week for weekly trading...")
    
    weekly_results = []
    
    for day_offset in range(7):
        for min_spread in [0, 20, 50]:
            result_df = simulate_weekly_strategy(signals_df, min_spread, day_offset)
            
            if len(result_df) == 0:
                continue
            
            total_pnl = result_df['cumulative_pnl'].iloc[-1] if len(result_df) > 0 else 0
            num_trades = len(result_df[result_df['pnl'] != 0])
            avg_pnl = result_df[result_df['pnl'] != 0]['pnl'].mean() if num_trades > 0 else 0
            win_rate = (result_df[result_df['pnl'] > 0].shape[0] / num_trades * 100) if num_trades > 0 else 0
            
            weekly_results.append({
                'day_offset': day_offset,
                'min_spread': min_spread,
                'total_pnl': total_pnl * 100,
                'num_trades': num_trades,
                'avg_pnl': avg_pnl * 100,
                'win_rate': win_rate
            })
    
    weekly_df = pd.DataFrame(weekly_results)
    
    # Show results by spread threshold
    for spread in [0, 20, 50]:
        print(f"\n📊 Weekly Trading - Spread ≥{spread} points:")
        subset = weekly_df[weekly_df['min_spread'] == spread].sort_values('total_pnl', ascending=False)
        
        if len(subset) == 0:
            print("  No trades met this threshold")
            continue
        
        print(f"\n{'Day Offset':>12s} | {'Trades':>7s} | {'Total PnL':>10s} | {'Avg PnL':>9s} | {'Win Rate':>8s}")
        print("-" * 60)
        for _, row in subset.iterrows():
            print(f"{'Day ' + str(int(row['day_offset'])):>12s} | {row['num_trades']:>7.0f} | {row['total_pnl']:>9.2f}% | {row['avg_pnl']:>8.3f}% | {row['win_rate']:>7.1f}%")
        
        best = subset.iloc[0]
        print(f"\n  🏆 Best: Day {int(best['day_offset'])} - {best['total_pnl']:.2f}% total PnL, {best['num_trades']:.0f} trades")
    
    # Compare daily vs weekly
    print("\n" + "=" * 80)
    print("⚖️  DAILY vs WEEKLY COMPARISON")
    print("=" * 80)
    
    # Daily with threshold 20+, hold position
    daily_20 = strategy_df[strategy_df['strategy'] == 'Threshold 20+ (Hold on Low Confidence)'].iloc[0]
    
    # Daily with threshold 50+, hold position
    daily_50 = strategy_df[strategy_df['strategy'] == 'Threshold 50+ (Hold on Low Confidence)'].iloc[0]
    
    # Best weekly with threshold 20+
    weekly_20_best = weekly_df[weekly_df['min_spread'] == 20].nlargest(1, 'total_pnl').iloc[0] if len(weekly_df[weekly_df['min_spread'] == 20]) > 0 else None
    
    # Best weekly with threshold 50+
    weekly_50_best = weekly_df[weekly_df['min_spread'] == 50].nlargest(1, 'total_pnl').iloc[0] if len(weekly_df[weekly_df['min_spread'] == 50]) > 0 else None
    
    print(f"\n📊 Spread ≥20 points:")
    print(f"  Daily (Hold Position):  {daily_20['total_pnl']:>8.2f}% PnL | {daily_20['num_trades']:>5.0f} trades | {daily_20['win_rate']:>5.1f}% win rate")
    if weekly_20_best is not None:
        print(f"  Weekly (Best Day):      {weekly_20_best['total_pnl']:>8.2f}% PnL | {weekly_20_best['num_trades']:>5.0f} trades | {weekly_20_best['win_rate']:>5.1f}% win rate")
        diff = daily_20['total_pnl'] - weekly_20_best['total_pnl']
        print(f"  → Daily is {diff:+.2f}% better" if diff > 0 else f"  → Weekly is {-diff:+.2f}% better")
    
    print(f"\n📊 Spread ≥50 points:")
    print(f"  Daily (Hold Position):  {daily_50['total_pnl']:>8.2f}% PnL | {daily_50['num_trades']:>5.0f} trades | {daily_50['win_rate']:>5.1f}% win rate")
    if weekly_50_best is not None:
        print(f"  Weekly (Best Day):      {weekly_50_best['total_pnl']:>8.2f}% PnL | {weekly_50_best['num_trades']:>5.0f} trades | {weekly_50_best['win_rate']:>5.1f}% win rate")
        diff = daily_50['total_pnl'] - weekly_50_best['total_pnl']
        print(f"  → Daily is {diff:+.2f}% better" if diff > 0 else f"  → Weekly is {-diff:+.2f}% better")
    
    # Key insights
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)
    
    print("\n1️⃣  HOLDING POSITION vs CLOSING:")
    hold_20 = strategy_df[strategy_df['strategy'] == 'Threshold 20+ (Hold on Low Confidence)'].iloc[0]
    close_20 = strategy_df[strategy_df['strategy'] == 'Threshold 20+ (Close on Low Confidence)'].iloc[0]
    print(f"   Threshold 20+ with Hold:  {hold_20['total_pnl']:>7.2f}% PnL")
    print(f"   Threshold 20+ with Close: {close_20['total_pnl']:>7.2f}% PnL")
    print(f"   → Holding is {hold_20['total_pnl'] - close_20['total_pnl']:+.2f}% better")
    
    print("\n2️⃣  BEST OVERALL STRATEGY:")
    best_strategy = strategy_df.nlargest(1, 'total_pnl').iloc[0]
    print(f"   {best_strategy['strategy']}")
    print(f"   Total PnL: {best_strategy['total_pnl']:.2f}%")
    print(f"   Trades: {best_strategy['num_trades']:.0f}")
    print(f"   Win Rate: {best_strategy['win_rate']:.1f}%")
    
    print("\n3️⃣  WEEKLY TRADING:")
    if len(weekly_df) > 0:
        best_weekly = weekly_df.nlargest(1, 'total_pnl').iloc[0]
        print(f"   Best weekly strategy: Spread ≥{best_weekly['min_spread']:.0f}, Day {int(best_weekly['day_offset'])}")
        print(f"   Total PnL: {best_weekly['total_pnl']:.2f}%")
        print(f"   Trades: {best_weekly['num_trades']:.0f}")
        print(f"   Win Rate: {best_weekly['win_rate']:.1f}%")
    
    # Save results
    strategy_df.to_csv('threshold_strategy_results.csv', index=False)
    weekly_df.to_csv('weekly_strategy_results.csv', index=False)
    
    print("\n💾 Results saved to:")
    print("   - threshold_strategy_results.csv")
    print("   - weekly_strategy_results.csv")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
