#!/usr/bin/env python
"""
CryptoPricePredictor.py
-----------------------
DISCLAIMER: This script is an educational example only. It is not intended for use in live trading without extensive testing, 
validation, and professional review. Price predictions are based on historical data and machine learning models, which 
do not guarantee future results. Use at your own risk and do not rely on this for real financial decisions.

A cryptocurrency trading tool focused on predicting price levels using technical indicators and machine learning.
This script fetches daily historical data from Binance via CCXT for the last 180 days, calculates SMA and RSI using the ta library,
identifies one support and one resistance level as recent swing high/low over 20 days, and predicts future prices
using an enhanced LSTM model with fine-tuned scaling and adjustments.

Requirements:
- pip install ccxt pandas numpy matplotlib scipy ta scikit-learn requests tqdm tensorflow
- Python 3.7+

Installation Details:
- ccxt: Cryptocurrency exchange API wrapper
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib: Plotting library
- scipy: Scientific computing for peak detection
- ta: Technical analysis indicators
- scikit-learn: Machine learning tools (for preprocessing)
- requests: HTTP requests library (used by ccxt)
- tqdm: Progress bar for predictions
- tensorflow: Machine learning framework for LSTM model

Usage:
- Run the script: python CryptoPricePredictor.py
- Follow the prompts to enter a symbol (e.g., BTC, ETH, SOL) and number of days to predict
"""

import pandas as pd
import numpy as np
import ccxt
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def identify_support_resistance(data, lookback=20):
    """Identify support and resistance levels from recent swing highs/lows."""
    recent_data = data.tail(lookback)
    latest_price = recent_data['Close'].iloc[-1]
    
    peaks, _ = find_peaks(recent_data['Close'], distance=5)
    troughs, _ = find_peaks(-recent_data['Close'], distance=5)
    
    support = recent_data['Close'].iloc[troughs[-1]] if len(troughs) > 0 else np.nan
    resistance = recent_data['Close'].iloc[peaks[-1]] if len(peaks) > 0 else np.nan
    
    if pd.notna(support) and support >= latest_price:
        support = np.nan
    if pd.notna(resistance) and resistance <= latest_price:
        resistance = np.nan
    
    return support, resistance

def prepare_lstm_data(data, look_back=60):
    """Prepare data for LSTM model with multiple features."""
    features = ['Close', 'SMA', 'RSI', 'Trend', 'Volatility', 'Volume']
    data_clean = data[features].dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_clean)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val, scaler, data_clean.columns

def ml_price_prediction(data, days_ahead=7, sma_period=20, rsi_period=14, look_back=60):
    """Predict future prices using an LSTM model."""
    data['Trend'] = data['Close'].diff().rolling(window=5).mean()
    sma_indicator = SMAIndicator(close=data['Close'], window=sma_period)
    data['SMA'] = sma_indicator.sma_indicator()
    rsi_indicator = RSIIndicator(close=data['Close'], window=rsi_period)
    data['RSI'] = rsi_indicator.rsi()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    X_train, y_train, X_val, y_val, scaler, feature_names = prepare_lstm_data(data, look_back)
    
    model = Sequential()
    model.add(Input(shape=(look_back, X_train.shape[2])))
    model.add(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))
    model.add(LSTM(75, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    
    last_sequence = scaler.transform(data[feature_names].tail(look_back))
    last_sequence = last_sequence.reshape((1, look_back, len(feature_names)))
    
    predicted_prices = []
    current_sequence = last_sequence.copy()
    recent_closes = list(data['Close'].tail(rsi_period))
    latest_price = data['Close'].iloc[-1]
    min_price = latest_price * 0.9  # ±10% bounds
    max_price = latest_price * 1.1
    
    with tqdm(total=days_ahead, desc="Predicting prices") as pbar:
        for _ in range(days_ahead):
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            raw_scaled_price = scaler.inverse_transform(
                np.concatenate([[next_pred], current_sequence[0, -1, 1:]]).reshape(1, -1)
            )[0, 0]
            
            predicted_price = latest_price + (raw_scaled_price - latest_price) * 0.9
            current_sma = scaler.inverse_transform(current_sequence[0])[-1, 1]
            sma_factor = (current_sma - latest_price) * 0.03
            predicted_price += sma_factor
            
            predicted_price = min(max(predicted_price, min_price), max_price)
            
            recent_closes.append(predicted_price)
            recent_closes.pop(0)
            delta = pd.Series(recent_closes).diff()
            gain = delta.where(delta > 0, 0).mean()
            loss = -delta.where(delta < 0, 0).mean()
            new_rsi = 100 - (100 / (1 + gain/loss)) if loss != 0 else 100
            new_sma = (current_sma * sma_period + predicted_price - scaler.inverse_transform(current_sequence[0])[0, 0]) / sma_period
            new_trend = (predicted_price - scaler.inverse_transform([[current_sequence[0, -1, 0], 0, 0, 0, 0, 0]])[0, 0]) / 5
            new_volatility = current_sequence[0, :, 4].mean()
            new_volume = current_sequence[0, :, 5].mean()
            
            new_row = scaler.transform(
                np.array([predicted_price, new_sma, new_rsi, new_trend, new_volatility, new_volume]).reshape(1, -1)
            )[0]
            
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_row
            
            predicted_prices.append(predicted_price)
            pbar.update(1)
    
    return predicted_prices

def fetch_crypto_data(symbol, timeframe='1d', limit=180):
    """Fetch historical OHLCV data from Binance."""
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = f"{symbol.upper()}/USDT"  # Auto-append /USDT
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        data = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
        data.set_index('Timestamp', inplace=True)
        print(f"Fetched {len(data)} days of data for {symbol}")
        return data
    except ccxt.ExchangeError as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def analyze_crypto(symbol, sma_period=20, rsi_period=14, prediction_days=7):
    """Analyze and predict prices for a given cryptocurrency symbol."""
    symbol = symbol.upper()
    print(f"\nAnalyzing {symbol}/USDT...")
    
    data = fetch_crypto_data(symbol)
    if data.empty:
        print(f"No data available for {symbol}/USDT")
        return
    
    sma_indicator = SMAIndicator(close=data['Close'], window=sma_period)
    data['SMA'] = sma_indicator.sma_indicator()
    rsi_indicator = RSIIndicator(close=data['Close'], window=rsi_period)
    data['RSI'] = rsi_indicator.rsi()
    
    support, resistance = identify_support_resistance(data)
    
    predicted_prices = ml_price_prediction(data, prediction_days, sma_period, rsi_period)
    
    latest_close = data['Close'].iloc[-1]
    latest_sma = data['SMA'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    
    print(f"Latest Close: ${latest_close:,.4f}")
    print(f"SMA ({sma_period}-day): ${latest_sma:,.4f}")
    print(f"RSI ({rsi_period}-day): {latest_rsi:.2f}")
    
    print("\nSupport Level (Recent Swing Low, last 20 days):")
    print(f"  ${support:,.4f}" if pd.notna(support) else "  No recent support level found")
    
    print("\nResistance Level (Recent Swing High, last 20 days):")
    print(f"  ${resistance:,.4f}" if pd.notna(resistance) else "  No recent resistance level found")
    
    print(f"\nPredicted Prices for Next {prediction_days} Days:")
    for i, price in enumerate(predicted_prices, 1):
        print(f"  Day {i}: ${price:,.4f}")

def main():
    """Run the script with interactive user inputs."""
    print("Welcome to CryptoPricePredictor!")
    
    # Get symbol from user
    while True:
        symbol = input("Enter cryptocurrency symbol (e.g., BTC, ETH, SOL): ").strip().upper()
        if symbol:
            break
        print("Symbol cannot be empty. Please try again.")
    
    # Get prediction days from user
    while True:
        days_input = input("Enter number of days to predict (default is 7): ").strip()
        if not days_input:  # If empty, use default
            prediction_days = 7
            break
        try:
            prediction_days = int(days_input)
            if prediction_days > 0:
                break
            print("Number of days must be positive. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nStarting analysis for {symbol}/USDT with {prediction_days} days prediction...")
    analyze_crypto(symbol, sma_period=20, rsi_period=14, prediction_days=prediction_days)

if __name__ == "__main__":
    main()