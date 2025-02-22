# CryptoSRPredictor.py
# A cryptocurrency trading tool focused on predicting support and resistance levels
# using technical indicators and machine learning

import ccxt              # For fetching cryptocurrency exchange data
import pandas as pd      # For data manipulation and analysis
import numpy as np       # For numerical operations
import ta                # For technical analysis indicators
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.ensemble import RandomForestClassifier   # For machine learning classification
from sklearn.metrics import accuracy_score           # For evaluating model performance
import tkinter as tk     # For GUI creation
from tkinter import ttk, scrolledtext  # For GUI widgets

# Function to perform support/resistance analysis
def run_analysis(symbol, timeframe, limit):
    """Analyze market data and predict support/resistance levels"""
    exchange_name = 'binance'
    output_text.delete(1.0, tk.END)  # Clear previous output in GUI

    # Fetch market data using CCXT (no API key needed for public data)
    try:
        exchange = getattr(ccxt, exchange_name)()
        exchange.load_markets()
        if symbol not in exchange.symbols:
            output_text.insert(tk.END, f"Error: {symbol} is not available on {exchange_name}.\n")
            return

        output_text.insert(tk.END, f"Fetching {limit} candles for {symbol} from {exchange_name}...\n")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            output_text.insert(tk.END, "Error: No OHLCV data received. Check symbol and timeframe.\n")
            return
    except Exception as e:
        output_text.insert(tk.END, f"Error fetching data: {e}\n")
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

    # Predict current market condition
    last_row = pd.DataFrame([X.iloc[-1]], columns=feature_columns)
    is_support = support_model.predict(last_row)[0]
    is_resistance = resistance_model.predict(last_row)[0]

    # Get key levels
    nearest_support = df['Support1'].iloc[-1]
    nearest_resistance = df['Resistance1'].iloc[-1]
    current_price = df['Close'].iloc[-1]

    # Display results in GUI
    output_text.insert(tk.END, "\n📊 **Updated Model Accuracy Metrics:**\n")
    output_text.insert(tk.END, f"   ✅ Support Level Accuracy: {support_accuracy:.4f}\n")
    output_text.insert(tk.END, f"   ✅ Resistance Level Accuracy: {resistance_accuracy:.4f}\n\n")

    output_text.insert(tk.END, f"📈 **Current Market Condition for {symbol} on {exchange_name}:**\n")
    output_text.insert(tk.END, f"   Current Price: {current_price:.4f}\n")
    if is_support:
        output_text.insert(tk.END, f"   ✅ The current price is at a **Support Level**! (Potential Buy Zone) 🟢\n")
    if is_resistance:
        output_text.insert(tk.END, f"   ❌ The current price is at a **Resistance Level**! (Potential Sell Zone) 🔴\n")
    if not is_support and not is_resistance:
        output_text.insert(tk.END, f"   ⚖️ The price is in a neutral zone (no clear support/resistance).\n")

    output_text.insert(tk.END, "\n📌 **Nearest Key Levels:**\n")
    output_text.insert(tk.END, f"   🟢 Nearest Support Level: {nearest_support:.4f}\n")
    output_text.insert(tk.END, f"   🔴 Nearest Resistance Level: {nearest_resistance:.4f}\n")

# GUI Setup with Dark Theme
root = tk.Tk()
root.title("Crypto Support/Resistance Predictor")
root.geometry("900x500")
root.configure(bg='#2b2b2b')  # Set dark background

# Define dark theme style
style = ttk.Style()
style.theme_use('clam')  # Use 'clam' theme as base for customization
style.configure('TButton', background='#404040', foreground='white', bordercolor='#555555')
style.configure('TLabel', background='#2b2b2b', foreground='white')

# Create input fields with dark theme
tk.Label(root, text="Trading Pair (e.g., BTC/USDT):", bg='#2b2b2b', fg='white').pack(pady=5)
symbol_entry = tk.Entry(root, bg='#404040', fg='white', insertbackground='white')  # White cursor
symbol_entry.pack()

tk.Label(root, text="Timeframe (e.g., 1d, 1h, 4h):", bg='#2b2b2b', fg='white').pack(pady=5)
timeframe_entry = tk.Entry(root, bg='#404040', fg='white', insertbackground='white')
timeframe_entry.pack()

tk.Label(root, text="Number of Candles (e.g., 2000):", bg='#2b2b2b', fg='white').pack(pady=5)
limit_entry = tk.Entry(root, bg='#404040', fg='white', insertbackground='white')
limit_entry.pack()

# Start button with dark theme
start_button = ttk.Button(root, text="Start Analysis", command=lambda: run_analysis(
    symbol_entry.get().strip().upper(),
    timeframe_entry.get().strip(),
    int(limit_entry.get())
))
start_button.pack(pady=10)

# Output display area with dark theme
output_text = scrolledtext.ScrolledText(root, width=70, height=20, bg='#333333', fg='white',
                                       insertbackground='white')  # White cursor
output_text.pack(pady=10)

# Start the GUI event loop
root.mainloop()

# To run this script:
# 1. Save as CryptoSRPredictor.py
# 2. Ensure all dependencies are installed (see requirements.txt)
# 3. Run with: python CryptoSRPredictor.py
# 4. Enter trading pair, timeframe, and candle count in the GUI and click "Start Analysis"