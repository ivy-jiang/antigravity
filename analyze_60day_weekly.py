import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
60-DAY WEEKLY TRADING ANALYSIS
Tests weekly trading strategy on recent 60 days with proper day-of-week alignment.
"""

def load_model(model_path, state_dim, action_dim):
    """Loads the trained model."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_all_signals(model, data, window_size=20):
    """Gets all trading signals with Q-values and spreads."""
    env = FinancialEnv(data, window_size=window_size)
    
    results = []
    state = env.reset()
    done = False
    step = 0
    
    prices = data['price'].values
    
    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)[0].numpy()
            
            sorted_q = np.sort(q_values)[::-1]
            spread = sorted_q[0] - sorted_q[1]
            action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            
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

def simulate_weekly_trading(signals_df, day_offset=0, min_spread=0):
    """
    Simulates weekly trading starting from a specific day offset.
    
    Args:
        signals_df: Daily signals DataFrame
        day_offset: Which day to start (0-6)
        min_spread: Minimum spread threshold (0 = take all signals)
    
    Returns:
        DataFrame with trade results
    """
    action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    
    results = []
    current_position = 0  # 0=cash, 1=long, -1=short
    entry_price = 0
    cumulative_pnl = 0
    
    # Get weekly signals (every 7 days starting from day_offset)
    weekly_indices = list(range(day_offset, len(signals_df), 7))
    
    for idx in weekly_indices:
        if idx >= len(signals_df):
            break
        
        row = signals_df.iloc[idx]
        step = row['step']
        signal_action = row['action']
        spread = row['spread']
        price = row['price']
        
        # Check if signal meets threshold
        if spread < min_spread:
            # Skip this trade
            results.append({
                'step': step,
                'action': 'Skip',
                'reason': f'Spread {spread:.2f} < {min_spread}',
                'spread': spread,
                'position': current_position,
                'price': price,
                'pnl': 0,
                'cumulative_pnl': cumulative_pnl
            })
            continue
        
        actual_action = signal_action
        pnl = 0
        
        # Execute trade
        if actual_action == 1:  # Buy
            if current_position == 0:
                current_position = 1
                entry_price = price
                action_desc = 'Enter Long'
            elif current_position == -1:
                pnl = (entry_price - price) / entry_price
                current_position = 1
                entry_price = price
                action_desc = 'Close Short, Enter Long'
            else:
                action_desc = 'Hold Long'
        
        elif actual_action == 2:  # Sell
            if current_position == 0:
                current_position = -1
                entry_price = price
                action_desc = 'Enter Short'
            elif current_position == 1:
                pnl = (price - entry_price) / entry_price
                current_position = -1
                entry_price = price
                action_desc = 'Close Long, Enter Short'
            else:
                action_desc = 'Hold Short'
        
        elif actual_action == 0:  # Hold
            action_desc = 'Hold'
        
        cumulative_pnl += pnl
        
        results.append({
            'step': step,
            'action': action_names[actual_action],
            'action_desc': action_desc,
            'spread': spread,
            'position': current_position,
            'price': price,
            'pnl': pnl * 100,  # Convert to percentage
            'cumulative_pnl': cumulative_pnl * 100
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("=" * 80)
    print("📊 60-DAY WEEKLY TRADING ANALYSIS")
    print("=" * 80)
    
    # Configuration
    data_file = 'qqq_data_60days.csv'
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    # Today's info
    today = datetime(2024, 11, 20)  # Thursday
    print(f"\n📅 Today: {today.strftime('%A, %B %d, %Y')} (Thursday)")
    
    # Load data
    print(f"\n📊 Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    print(f"✅ Data loaded: {len(df)} days")
    print(f"   Date range: Last {len(df)} days ending Nov 19, 2024")
    
    # Apply preprocessing
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    feature_cols = [c for c in df.columns if c != 'price']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    # Load model
    print(f"\n🧠 Loading model...")
    temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
    agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
    print(f"✅ Model loaded")
    
    # Get all signals
    print(f"\n🔍 Generating signals for all {len(df)} days...")
    signals_df = get_all_signals(agent, df, WINDOW_SIZE)
    print(f"✅ Generated {len(signals_df)} signals")
    
    # Figure out day alignment
    print("\n" + "=" * 80)
    print("📅 DAY OF WEEK ALIGNMENT")
    print("=" * 80)
    
    # If today (Nov 20) is Thursday and is the last data point
    # Then the data spans from Aug 28 to Nov 19 (60 days)
    # Aug 28, 2024 was a Wednesday
    # So Day 0 = Aug 28 (Wednesday)
    
    start_date = datetime(2024, 8, 28)  # First day in dataset
    end_date = datetime(2024, 11, 19)   # Last day in dataset
    
    print(f"\n📊 Data Period:")
    print(f"   Start: {start_date.strftime('%A, %B %d, %Y')} (Day 0)")
    print(f"   End:   {end_date.strftime('%A, %B %d, %Y')} (Day {(end_date - start_date).days})")
    print(f"   Total: {(end_date - start_date).days + 1} days")
    
    # Map day offsets to actual days of week
    print(f"\n📅 Day Offset Mapping:")
    for offset in range(7):
        actual_date = start_date + timedelta(days=offset)
        print(f"   Day {offset}: {actual_date.strftime('%A')}")
    
    # Day 0 = Wednesday, so Day 5 = Monday
    print(f"\n🎯 Key Finding:")
    print(f"   Day 5 = Monday (best performing day in historical analysis)")
    print(f"   Day 2 = Friday (best for spread ≥20 strategy)")
    
    # Test weekly strategies
    print("\n" + "=" * 80)
    print("🧪 TESTING WEEKLY STRATEGIES ON 60-DAY DATA")
    print("=" * 80)
    
    strategies = [
        (0, 0, "All Signals"),
        (20, 0, "Spread ≥20"),
        (50, 0, "Spread ≥50"),
    ]
    
    all_results = []
    
    for min_spread, _, strategy_name in strategies:
        print(f"\n📊 Strategy: {strategy_name}")
        print("-" * 80)
        
        for day_offset in range(7):
            day_name = (start_date + timedelta(days=day_offset)).strftime('%A')
            
            result_df = simulate_weekly_trading(signals_df, day_offset, min_spread)
            
            if len(result_df) == 0:
                continue
            
            # Calculate metrics
            trades_executed = len(result_df[result_df['action'] != 'Skip'])
            total_pnl = result_df['cumulative_pnl'].iloc[-1] if len(result_df) > 0 else 0
            
            # Count profitable trades
            profitable = len(result_df[result_df['pnl'] > 0])
            losing = len(result_df[result_df['pnl'] < 0])
            win_rate = (profitable / trades_executed * 100) if trades_executed > 0 else 0
            
            all_results.append({
                'strategy': strategy_name,
                'min_spread': min_spread,
                'day_offset': day_offset,
                'day_name': day_name,
                'total_trades': len(result_df),
                'executed_trades': trades_executed,
                'total_pnl': total_pnl,
                'win_rate': win_rate
            })
            
            print(f"  Day {day_offset} ({day_name:9s}): {trades_executed:2d} trades | {total_pnl:>8.2f}% PnL | {win_rate:>5.1f}% win rate")
    
    # Show best strategies
    print("\n" + "=" * 80)
    print("🏆 BEST STRATEGIES FOR 60-DAY PERIOD")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    for strategy_name in ["All Signals", "Spread ≥20", "Spread ≥50"]:
        subset = results_df[results_df['strategy'] == strategy_name]
        if len(subset) == 0:
            continue
        
        best = subset.nlargest(1, 'total_pnl').iloc[0]
        
        print(f"\n📊 {strategy_name}:")
        print(f"   Best Day: Day {int(best['day_offset'])} ({best['day_name']})")
        print(f"   Trades: {int(best['executed_trades'])}")
        print(f"   Total PnL: {best['total_pnl']:.2f}%")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
    
    # Current recommendation
    print("\n" + "=" * 80)
    print("📢 CURRENT RECOMMENDATION (Nov 20, 2024)")
    print("=" * 80)
    
    # Get latest signal
    latest_signal = signals_df.iloc[-1]
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    print(f"\n🔮 Latest Signal (Nov 19 data):")
    print(f"   Action: {action_names[latest_signal['action']]}")
    print(f"   Spread: {latest_signal['spread']:.2f} points")
    print(f"   Q-Values: Hold={latest_signal['q_hold']:.2f}, Buy={latest_signal['q_buy']:.2f}, Sell={latest_signal['q_sell']:.2f}")
    
    # Determine what day we're on
    days_since_start = (today - start_date).days
    current_day_offset = days_since_start % 7
    current_day_name = today.strftime('%A')
    
    print(f"\n📅 Today's Position in Cycle:")
    print(f"   Today: {current_day_name} (Nov 20)")
    print(f"   Days since start: {days_since_start}")
    print(f"   Current day offset: {current_day_offset}")
    
    # Find when Day 5 (Monday) occurs
    day_5_name = (start_date + timedelta(days=5)).strftime('%A')
    days_until_day_5 = (5 - current_day_offset) % 7
    next_day_5 = today + timedelta(days=days_until_day_5)
    
    print(f"\n🎯 Next Trading Day (Day 5 = {day_5_name}):")
    print(f"   Date: {next_day_5.strftime('%A, %B %d, %Y')}")
    print(f"   Days from now: {days_until_day_5}")
    
    # Trading rule
    print("\n" + "=" * 80)
    print("📋 TRADING RULE TO FOLLOW")
    print("=" * 80)
    
    print(f"\n✅ RECOMMENDED STRATEGY: Weekly Trading on Day 5 ({day_5_name})")
    print(f"\n📅 Schedule:")
    print(f"   Trade every {day_5_name}")
    print(f"   Next trade: {next_day_5.strftime('%B %d, %Y')}")
    print(f"\n📝 Process:")
    print(f"   1. Every {day_5_name}, run: python3 fetch_qqq.py")
    print(f"   2. Run: python3 get_recommendation.py")
    print(f"   3. Take the action (Buy/Sell/Hold) REGARDLESS of spread/confidence")
    print(f"   4. Hold position for 7 days")
    print(f"   5. Repeat next {day_5_name}")
    
    print(f"\n🎯 Key Points:")
    print(f"   • NO threshold needed - take all signals")
    print(f"   • Trade ONLY on {day_5_name}s")
    print(f"   • Ignore signals on other days")
    print(f"   • Based on 289% PnL in historical analysis")
    
    if days_until_day_5 == 0:
        print(f"\n🚨 TODAY IS TRADING DAY!")
        print(f"   Current signal: {action_names[latest_signal['action']]}")
        print(f"   → Execute this trade today!")
    else:
        print(f"\n⏳ NOT A TRADING DAY")
        print(f"   Wait {days_until_day_5} days until {next_day_5.strftime('%A, %B %d')}")
        print(f"   Current signal ({action_names[latest_signal['action']]}) is for reference only")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
