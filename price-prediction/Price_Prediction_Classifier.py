"""
Price_Prediction_Classifier.py
------------------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script fetches historical OHLCV data from Binance using CCXT, calculates RSI and MACD indicators using TA-Lib,
and uses a RandomForestClassifier to predict whether the price will go up (Buy) or down (Sell) in the next period.
The user can dynamically input the cryptocurrency symbol (e.g., BTC), timeframe (e.g., 1h), and data limit after running the script.

Requirements:
- pip install ccxt pandas numpy sklearn talib
- Python 3.7+
- TA-Lib binary installed (see: https://github.com/TA-Lib/ta-lib-python#dependencies)

Usage:
- Run the script: python Price_Prediction_Classifier.py
- Follow the prompts to enter:
  - Cryptocurrency symbol (e.g., BTC, ETH, SOL)
  - Timeframe (e.g., 1m, 1h, 1d)
  - Number of periods to fetch (e.g., 600)
"""

import pandas as pd
import numpy as np
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import talib

def fetch_data(symbol, timeframe, limit):
    """Fetch OHLCV data from Binance using CCXT."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    try:
        symbol = f"{symbol.upper()}/USDT"
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        print(f"Fetched {len(data)} periods of data for {symbol} at {timeframe} timeframe")
        return data
    except ccxt.ExchangeError as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(data):
    """Calculate RSI and MACD indicators using TA-Lib with correct alignment."""
    close_prices = data['close'].values
    assert close_prices.ndim == 1, "close_prices array is not one-dimensional."
    print(f"Close prices length: {len(close_prices)}")

    # Initialize columns with NaN
    data['RSI'] = np.nan
    data['MACD'] = np.nan
    data['Signal'] = np.nan

    # Calculate RSI (14-period)
    rsi = talib.RSI(close_prices, timeperiod=14)
    print(f"RSI length: {len(rsi)}")  # Debug output
    rsi_start_idx = 14
    rsi_end_idx = rsi_start_idx + len(rsi)
    if rsi_end_idx <= len(data):
        data.iloc[rsi_start_idx:rsi_end_idx, data.columns.get_loc('RSI')] = rsi
    else:
        # If RSI is longer, trim it to fit (shouldn't happen, but handles edge case)
        valid_rsi = rsi[:len(data) - rsi_start_idx]
        data.iloc[rsi_start_idx:rsi_start_idx + len(valid_rsi), data.columns.get_loc('RSI')] = valid_rsi
        print(f"Trimmed RSI to fit: {len(valid_rsi)} values assigned")

    # Calculate MACD (12, 26, 9)
    macd, macd_signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    print(f"MACD length: {len(macd)}, Signal length: {len(macd_signal)}")  # Debug output
    macd_start_idx = 33  # 26 (slow EMA) + 9 (signal) - 1
    macd_end_idx = macd_start_idx + len(macd)
    if macd_end_idx <= len(data):
        data.iloc[macd_start_idx:macd_end_idx, data.columns.get_loc('MACD')] = macd
        data.iloc[macd_start_idx:macd_end_idx, data.columns.get_loc('Signal')] = macd_signal
    else:
        # Trim MACD to fit if longer (handles unexpected behavior)
        valid_macd = macd[:len(data) - macd_start_idx]
        valid_signal = macd_signal[:len(data) - macd_start_idx]
        data.iloc[macd_start_idx:macd_start_idx + len(valid_macd), data.columns.get_loc('MACD')] = valid_macd
        data.iloc[macd_start_idx:macd_start_idx + len(valid_signal), data.columns.get_loc('Signal')] = valid_signal
        print(f"Trimmed MACD to fit: {len(valid_macd)} values assigned")

    return data

def prepare_data(data):
    """Prepare features and target for classification."""
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
    features = ['RSI', 'MACD', 'Signal']
    X = data[features].dropna()
    y = data['Target'].loc[X.index]
    print(f"Valid data points after NaN removal: {len(X)}")  # Debug output
    if X.empty:
        print("No valid data after dropping NaNs. Increase the limit (min 50 recommended).")
    return X, y

def train_model(X, y):
    """Train a RandomForestClassifier model."""
    if X.empty or y.empty:
        raise ValueError("Cannot train model: Feature set or target is empty.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2%}")
    return model

def generate_signals(data, model):
    """Generate buy/sell signals based on model predictions."""
    features = ['RSI', 'MACD', 'Signal']
    X = data[features].dropna()
    data['Prediction'] = np.nan
    data.loc[X.index, 'Prediction'] = model.predict(X)
    data['Buy'] = data['Prediction'] == 1
    data['Sell'] = data['Prediction'] == 0
    return data

def main():
    """Run the script with interactive user inputs."""
    print("Welcome to Price Prediction Classifier!")

    while True:
        symbol = input("Enter cryptocurrency symbol (e.g., BTC, ETH, SOL): ").strip().upper()
        if symbol:
            break
        print("Symbol cannot be empty. Please try again.")

    while True:
        timeframe = input("Enter timeframe (e.g., 1m, 5m, 1h, 1d): ").strip().lower()
        if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']:
            break
        print("Invalid timeframe. Examples: 1m, 5m, 1h, 1d. Please try again.")

    while True:
        limit_input = input("Enter number of periods to fetch (e.g., 600, min 50): ").strip()
        try:
            limit = int(limit_input)
            if limit >= 50:
                break
            print("Number of periods must be at least 50 for indicator warmup. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nFetching data for {symbol}/USDT at {timeframe} timeframe with {limit} periods...")
    data = fetch_data(symbol, timeframe, limit)
    if data.empty:
        print("No data fetched. Exiting.")
        return

    data = calculate_indicators(data)
    X, y = prepare_data(data)
    
    if X.empty:
        print("Exiting due to insufficient valid data after indicator calculation.")
        return

    try:
        model = train_model(X, y)
        data = generate_signals(data, model)
        print("\nLatest data with predictions:")
        print(data.tail()[['close', 'RSI', 'MACD', 'Signal', 'Prediction', 'Buy', 'Sell']])
    except ValueError as e:
        print(f"Error during modeling: {e}")

if __name__ == "__main__":
    main()