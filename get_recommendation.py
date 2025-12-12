import torch
import numpy as np
import pandas as pd
from prospect_theory_agent import ProspectTheoryQNetwork, FinancialEnv

"""
LIVE RECOMMENDATION SCRIPT
Get a BUY/SELL/HOLD recommendation for the current market state.
"""

def load_model(model_path, state_dim, action_dim):
    """Loads the trained model."""
    model = ProspectTheoryQNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_current_recommendation(model, data, window_size=20):
    """
    Gets a recommendation based on the MOST RECENT data.
    
    Args:
        model: Trained model
        data: DataFrame with market data
        window_size: How many days to look back
    
    Returns:
        action: 0=Hold, 1=Buy, 2=Sell
        confidence: Q-value for the chosen action
        q_values: All Q-values for transparency
    """
    # Get the most recent window_size days
    if len(data) < window_size:
        raise ValueError(f"Need at least {window_size} days of data, got {len(data)}")
    
    # Create environment to get proper state
    env = FinancialEnv(data, window_size=window_size)
    
    # Reset environment and step through to the last state
    state = env.reset()
    done = False
    
    # Step through all data to get to the final state
    while not done:
        # Take a dummy action (doesn't matter, we just want the final state)
        next_state, _, done = env.step(0)  # 0 = Hold
        if not done:
            state = next_state
    
    # Now 'state' contains the most recent market condition
    # Get model prediction
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        action = q_values.argmax(dim=1).item()
        confidence = q_values[0, action].item()
    
    return action, confidence, q_values[0].numpy()

def format_recommendation(action, confidence, q_values):
    """Formats the recommendation for display."""
    action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    action_emojis = {0: '⏸️', 1: '📈', 2: '📉'}
    
    recommendation = action_names[action]
    emoji = action_emojis[action]
    
    # Calculate confidence as percentage difference from other actions
    sorted_q = np.sort(q_values)[::-1]
    confidence_pct = ((sorted_q[0] - sorted_q[1]) / abs(sorted_q[0]) * 100) if sorted_q[0] != 0 else 0
    
    return recommendation, emoji, confidence_pct, q_values

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 LIVE TRADING RECOMMENDATION")
    print("=" * 70)
    
    # Configuration
    data_file = 'qqq_data_60days.csv'  # Using fresh 60-day data
    model_path = 'prospect_theory_model.pth'
    WINDOW_SIZE = 20
    
    # Load data
    print(f"\n📊 Loading market data from {data_file}...")
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
        print(f"📅 Latest date in dataset: Row {len(df)}")
        
    except FileNotFoundError:
        print(f"❌ Error: {data_file} not found.")
        exit()
    
    # Load model
    print(f"\n🧠 Loading trained model from {model_path}...")
    try:
        # Create a temporary environment to get dimensions
        temp_env = FinancialEnv(df, window_size=WINDOW_SIZE)
        agent = load_model(model_path, temp_env.state_dim, temp_env.action_dim)
        print(f"✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: {model_path} not found. Train the model first.")
        exit()
    
    # Get recommendation
    print(f"\n🔮 Analyzing current market conditions...")
    print(f"   Using last {WINDOW_SIZE} days of data...")
    
    action, confidence, q_values = get_current_recommendation(agent, df, WINDOW_SIZE)
    recommendation, emoji, confidence_pct, all_q_values = format_recommendation(action, confidence, q_values)
    
    # Display recommendation
    print("\n" + "=" * 70)
    print("📢 RECOMMENDATION")
    print("=" * 70)
    print(f"\n{emoji}  Action: {recommendation}")
    print(f"💪 Confidence: {abs(confidence_pct):.1f}%")
    
    print(f"\n📊 Q-Values (Model's Internal Scores):")
    print(f"   HOLD: {all_q_values[0]:.4f}")
    print(f"   BUY:  {all_q_values[1]:.4f}")
    print(f"   SELL: {all_q_values[2]:.4f}")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if recommendation == 'BUY':
        print(f"   The model expects prices to RISE based on recent patterns.")
        print(f"   It learned this pattern from historical data where similar")
        print(f"   conditions led to profitable buy opportunities.")
    elif recommendation == 'SELL':
        print(f"   The model expects prices to FALL based on recent patterns.")
        print(f"   It learned this pattern from historical data where similar")
        print(f"   conditions led to profitable sell opportunities.")
    else:
        print(f"   The model is UNCERTAIN or sees no clear opportunity.")
        print(f"   It's better to wait for a clearer signal.")
    
    if abs(confidence_pct) > 20:
        print(f"\n   ✅ HIGH CONFIDENCE - The model is quite sure about this decision.")
    elif abs(confidence_pct) > 10:
        print(f"\n   ⚠️  MODERATE CONFIDENCE - The model has some conviction.")
    else:
        print(f"\n   ⚠️  LOW CONFIDENCE - The model is uncertain. Be cautious.")
    
    # Important disclaimers
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT NOTES")
    print("=" * 70)
    print("1. This recommendation is based on HISTORICAL patterns")
    print("2. The model was trained on data ending at row", len(df))
    print("3. For CURRENT data, you need to update the CSV with today's data")
    print("4. This is NOT financial advice - use at your own risk")
    print("5. The model doesn't account for news, events, or fundamentals")
    print("=" * 70)
