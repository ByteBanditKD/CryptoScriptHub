# CryptoSRPredictor_CLI.py
# A cryptocurrency trading tool focused on predicting support and resistance levels
# using technical indicators and machine learning (CLI version)

import ccxt              # For fetching cryptocurrency exchange data
import pandas as pd      # For data manipulation and analysis
import numpy as np       # For numerical operations
import ta                # For technical analysis indicators
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.ensemble import RandomForestClassifier   # For machine learning classification
from sklearn.metrics import accuracy_score           # For evaluating model performance

# Function to perform support/resistance analysis
def run_analysis(symbol, timeframe, limit):
    """Analyze market data and predict support/resistance levels"""
    exchange_name = 'binance'

    # Fetch market data using CCXT (no API key needed for public data)
    try:
        exchange = getattr(ccxt, exchange_name)()
        exchange.load_markets()
        if symbol not in exchange.symbols:
            print(f"Error: {symbol} is not available on {exchange_name}.")
            return

        print(f"Fetching {limit} candles for {symbol} from {exchange_name}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            print("Error: No OHLCV data received. Check symbol and timeframe.")
            return
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Convert OHLCV data to DataFrame
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')  # Convert timestamp to datetime
    df.set_index('Date', inplace=True)

    # Calculate technical indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()    # 5-period Moving Average
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20-period Moving Average
    df['MA50'] = df['Close'].rolling(window=50).mean()  # 50-period Moving Average
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()  # Relative Strength Index
    df['MACD'] = ta.trend.MACD(df['Close']).macd()      # MACD line
    df['Signal_Line'] = ta.trend.MACD(df['Close']).macd_signal()  # MACD signal line
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()  # Average True Range
    df['Boll_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()  # Bollinger Band Upper
    df['Boll_Mid'] = ta.volatility.BollingerBands(df['Close']).bollinger_mavg()    # Bollinger Band Middle
    df['Boll_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()  # Bollinger Band Lower
    df['VWAP'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume'].rolling(window=14).sum() / 
                  df['Volume'].rolling(window=14).sum())  # Volume Weighted Average Price

    # Calculate Pivot Points for Support and Resistance
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Support1'] = df['Pivot'] - (df['High'] - df['Low'])
    df['Resistance1'] = df['Pivot'] + (df['High'] - df['Low'])

    # Identify Support/Resistance zones with buffer
    buffer_pct = 0.01  # 1% buffer for S/R identification
    df['Support'] = np.where((df['Low'] <= df['Support1'] * (1 + buffer_pct)), 1, 0)
    df['Resistance'] = np.where((df['High'] >= df['Resistance1'] * (1 - buffer_pct)), 1, 0)

    df.dropna(inplace=True)  # Remove rows with NaN values

    # Prepare features and labels for ML
    feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line', 
                      'ATR', 'Boll_Upper', 'Boll_Lower', 'VWAP', 'Pivot', 'Support1', 'Resistance1']
    X = df[feature_columns]
    y_support = df['Support']
    y_resistance = df['Resistance']

    # Split data into training and testing sets
    X_train_supp, X_test_supp, y_train_supp, y_test_supp = train_test_split(X, y_support, 
                                                                           test_size=0.2, random_state=42)
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X, y_resistance, 
                                                                        test_size=0.2, random_state=42)

    # Initialize and train Random Forest models
    support_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    resistance_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    support_model.fit(X_train_supp, y_train_supp)
    resistance_model.fit(X_train_res, y_train_res)

    # Evaluate model performance
    y_pred_supp = support_model.predict(X_test_supp)
    y_pred_res = resistance_model.predict(X_test_res)
    support_accuracy = accuracy_score(y_test_supp, y_pred_supp)
    resistance_accuracy = accuracy_score(y_test_res, y_pred_res)

    # Predict current market condition with probabilities
    last_row = pd.DataFrame([X.iloc[-1]], columns=feature_columns)
    support_prob = support_model.predict_proba(last_row)[0][1]  # Probability of being support
    resistance_prob = resistance_model.predict_proba(last_row)[0][1]  # Probability of being resistance

    # Determine the most likely condition (mutually exclusive)
    threshold = 0.5  # Probability threshold for classification
    if support_prob > resistance_prob and support_prob > threshold:
        market_condition = "support"
    elif resistance_prob > support_prob and resistance_prob > threshold:
        market_condition = "resistance"
    else:
        market_condition = "neutral"

    # Get key levels
    nearest_support = df['Support1'].iloc[-1]
    nearest_resistance = df['Resistance1'].iloc[-1]
    current_price = df['Close'].iloc[-1]

    # Display results in console
    print("\n📊 **Updated Model Accuracy Metrics:**")
    print(f"   ✅ Support Level Accuracy: {support_accuracy:.4f}")
    print(f"   ✅ Resistance Level Accuracy: {resistance_accuracy:.4f}\n")

    print(f"📈 **Current Market Condition for {symbol} on {exchange_name}:**")
    print(f"   Current Price: {current_price:.6f}")
    if market_condition == "support":
        print(f"   ✅ The current price is at a **Support Level**! (Potential Buy Zone) 🟢")
        print(f"   (Support Probability: {support_prob:.4f})")
    elif market_condition == "resistance":
        print(f"   ❌ The current price is at a **Resistance Level**! (Potential Sell Zone) 🔴")
        print(f"   (Resistance Probability: {resistance_prob:.4f})")
    else:
        print(f"   ⚖️ The price is in a neutral zone (no clear support/resistance).")
        print(f"   (Support Prob: {support_prob:.4f}, Resistance Prob: {resistance_prob:.4f})")

    print("\n📌 **Nearest Key Levels:**")
    print(f"   🟢 Nearest Support Level: {nearest_support:.6f}")
    print(f"   🔴 Nearest Resistance Level: {nearest_resistance:.6f}")

# Main execution
if __name__ == "__main__":
    # Get user input from command line
    symbol = input("Enter trading pair (e.g., BTC/USDT): ").strip().upper()
    timeframe = input("Enter timeframe (e.g., 1d, 1h, 4h): ").strip()
    limit = int(input("Enter number of candles (e.g., 2000): ").strip())

    # Run the analysis
    run_analysis(symbol, timeframe, limit)

# To run this script:
# 1. Save as CryptoSRPredictor_CLI.py
# 2. Ensure all dependencies are installed (see requirements.txt)
# 3. Run with: python CryptoSRPredictor_CLI.py
# 4. Follow the prompts to enter trading pair, timeframe, and candle count