#!/bin/bash

# Weekly Trading Assistant
# Run this on Thursday Night or Friday Morning

echo "========================================================"
echo "🚀 STARTING WEEKLY TRADING ANALYSIS"
echo "========================================================"
echo "📅 Date: $(date)"
echo ""

# 1. Fetch latest data
echo "📊 Step 1: Fetching latest QQQ data..."
/opt/anaconda3/bin/python3 fetch_qqq.py

if [ $? -ne 0 ]; then
    echo "❌ Error fetching data. Please check your internet connection."
    exit 1
fi
echo "✅ Data fetched successfully."
echo ""

# 2. Get Recommendation
echo "🧠 Step 2: Generating Trading Signal..."
echo "--------------------------------------------------------"
/opt/anaconda3/bin/python3 get_recommendation.py
echo "--------------------------------------------------------"

echo ""
echo "========================================================"
echo "📋 YOUR ACTION PLAN"
echo "========================================================"
echo "1. Read the RECOMMENDATION above (Buy/Sell/Hold)."
echo "2. If today is Thursday Night or Friday Morning:"
echo "   👉 Place this trade at FRIDAY MARKET OPEN."
echo "3. If today is Friday (Market Open):"
echo "   👉 Execute IMMEDIATELY."
echo "4. Hold this position for 7 days."
echo "========================================================"
