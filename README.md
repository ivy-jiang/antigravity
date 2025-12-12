# Antigravity Trading System 🚀

**Antigravity** is an AI-powered trading assistant designed to trade **QQQ** (Nasdaq-100 ETF) on a weekly basis. It uses **Prospect Theory** and **Reinforcement Learning** to make trading decisions that account for human-like risk perception (loss aversion).

## ⚡️ Quick Start: How to Trade

This system is designed for a **Weekly Trading Schedule** (typically Friday mornings).

1.  **Run the Assistant**:
    Open your terminal and run:
    ```bash
    ./weekly_trade.sh
    ```
2.  **Read the Signal**:
    The script will fetch the latest data and output a clear recommendation:
    *   📈 **BUY**: Open a long position.
    *   📉 **SELL**: Close position / Go short (depending on your strategy).
    *   ⏸️ **HOLD**: Do nothing.
3.  **Execute & Log**:
    *   Place the trade in your brokerage account.
    *   The system **automatically logs** this recommendation to `trade_log.csv`.

---

## 📂 Project Structure

### 🧠 Core System
*   **`weekly_trade.sh`**: The main entry point. Orchestrates data fetching and signal generation.
*   **`fetch_qqq.py`**: Fetches live market data from Yahoo Finance and FRED (Treasury yields). Generates `qqq_data_60days.csv`.
*   **`get_recommendation.py`**: Loads the trained AI model, analyzes the latest market state, prints the signal, and logs it to `trade_log.csv`.
*   **`prospect_theory_agent.py`**: Defines the Neural Network architecture (DQN) and the custom "Prospect Theory" loss function that makes the AI risk-aware.
*   **`trade_log.csv`**: Your official trading journal. Automatically updated with every signal.

### 📊 Analysis & Research
*   **`analyze_60day_weekly.py`**: Backtests the weekly strategy over the last 60 days.
*   **`analyze_advanced_strategies.py`**: Compares different trading rules (e.g., threshold-based trading vs. always trading).
*   **`analyze_confidence.py`**: Visualizes the model's confidence levels over time.
*   **`analyze_spread.py`**: Investigates the "spread" between Q-values (a measure of model certainty).
*   **`analyze_days.py`**: Determines which day of the week is statistically best for this strategy.

### 💾 Data & Models
*   **`prospect_theory_model.pth`**: The saved weights of the trained PyTorch model.
*   **`qqq_market_data.csv`**: Historical training data.
*   **`qqq_data_60days.csv`**: The most recent 60 days of data, used for live inference.

---

## 🛠 Setup & Requirements

### Prerequisites
*   Python 3.8+
*   `pip install pandas yfinance torch numpy pandas_datareader`

### GitHub Workflow
To save your changes and trade history to the cloud:

1.  **Save Changes**:
    ```bash
    git add .
    git commit -m "Weekly trade update"
    ```
2.  **Upload**:
    ```bash
    git push
    ```

---

## 🤖 How It Works
The agent observes a 20-day window of market features (Price, RSI, MACD, Treasury Yields, VIX). It calculates **Q-values** for Buying, Selling, or Holding. Unlike standard AI, it uses a **Prospect Theory** value function, meaning it is trained to be more sensitive to losses than gains, mimicking a disciplined human trader but with algorithmic precision.
