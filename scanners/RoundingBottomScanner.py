#!/usr/bin/env python
"""
RoundingBottomScanner.py
------------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for rounding bottom patterns using CCXT. It identifies pivot points using scipy.signal.argrelextrema,
detects parabolic rounding bottom formations with polynomial regression, generates candlestick charts with marked patterns, and saves results to a CSV file.
Users can choose to scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python RoundingBottomScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/rounding_bottom_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_rounding_bottom_[point].png'
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime
import mplfinance as mpf
from scipy.signal import argrelextrema
from tqdm import tqdm

# ================ CONFIGURATIONS ================= #
EXCHANGE_NAME = "bybit"
INTERVAL = "240"  # 4-hour interval
LIMIT = 50       # Number of candles to fetch
RESULTS_FOLDER = "results"
IMG_DIR = "img"
CSV_FILE = "pairs.csv"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"]
# ================================================== #

exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "future"}})

def adapt_symbol(symbol):
    """Convert symbol format to Bybit futures-compatible CCXT format (e.g., 'BTCUSDT' to 'BTC/USDT:USDT')."""
    symbol = symbol.strip().replace('\xa0', '')
    if '/' not in symbol:
        base = symbol[:-4].upper()
        quote = symbol[-4:].upper()
        return f"{base}/{quote}:{quote}"
    return symbol.upper()

def fetch_historical_data(symbol, interval=INTERVAL, limit=LIMIT):
    """Fetch historical OHLCV data for a given symbol from Bybit futures market."""
    exchange.load_markets()
    adapted_symbol = adapt_symbol(symbol)
    if adapted_symbol not in exchange.symbols:
        print(f"Error: {adapted_symbol} not available on {EXCHANGE_NAME} futures market. Skipping.")
        return None

    try:
        ohlcv = exchange.fetch_ohlcv(adapted_symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            print(f"Error: No data received for {adapted_symbol}.")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Remove duplicate timestamps, keeping the last entry
        df = df.drop_duplicates(subset='Date', keep='last')
        df.set_index('Date', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        if len(df) < limit:
            print(f"Warning: Insufficient data for {adapted_symbol}. Expected {limit}, got {len(df)}")
        
        return df
    except Exception as e:
        print(f"Error fetching data for {adapted_symbol}: {e}")
        return None

def find_rounding_bottom_points(ohlc, back_candles):
    """Find all rounding bottom points using polynomial regression."""
    all_points = []
    for candle_idx in range(back_candles + 10, len(ohlc)):
        minim = np.array([])
        xxmin = np.array([])

        for i in range(candle_idx - back_candles, candle_idx + 1):
            if ohlc.iloc[i]["Pivot"] == 1:
                minim = np.append(minim, ohlc.iloc[i]["close"])
                xxmin = np.append(xxmin, i)

        if len(xxmin) < 3:
            continue

        try:
            z = np.polyfit(xxmin, minim, 2)
        except np.linalg.LinAlgError:
            continue

        if 2 * z[0] > 0 and z[0] >= 2.19388889e-04 and z[1] <= -3.52871667e-02:
            all_points.append(candle_idx)

    return all_points

def generate_chart(ohlc, all_points, back_candles, symbol):
    """Generate and save candlestick chart with rounding bottom pattern."""
    results = []
    for point in all_points:
        start_idx = max(0, point - back_candles)
        if start_idx >= len(ohlc) or point + 1 > len(ohlc):
            print(f"Skipping point {point} for {symbol}: Index out of bounds (start_idx: {start_idx}, point: {point}, len: {len(ohlc)})")
            continue
        
        window_data = ohlc.iloc[start_idx:point + 1]
        if window_data.empty:
            print(f"Skipping point {point} for {symbol}: Empty window data")
            continue
        
        minim = np.array([])
        xxmin = np.array([])
        for i in range(start_idx, point + 1):
            if ohlc.iloc[i]["Pivot"] == 1:
                minim = np.append(minim, ohlc.iloc[i]["close"])
                xxmin = np.append(xxmin, i - start_idx)
        
        addplot = [mpf.make_addplot(window_data['PointPos'], type='scatter', markersize=100, 
                                  marker='v', color='blue', label='Pivot Lows')]
        
        if len(xxmin) >= 3:
            z = np.polyfit(xxmin, minim, 2)
            f = np.poly1d(z)
            x_fit = np.linspace(0, len(window_data) - 1, len(window_data))
            y_fit = f(x_fit)
            addplot.append(mpf.make_addplot(y_fit, type='line', color='orange', label='Rounding Bottom Fit'))

        img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_rounding_bottom_{point}.png")
        mpf.plot(window_data, type='candle', addplot=addplot, 
                title=f"Rounding Bottom for {symbol} at {window_data.index[-1]}",
                style='binance', savefig=img_path, ylabel='Price (USDT)')
        
        results.append({
            "Symbol": symbol,
            "Rounding Bottom Index": point,
            "Rounding Bottom Date": window_data.index[-1],
            "Current Price": round(window_data['close'].iloc[-1], 6),
            "Chart Path": img_path
        })
    return results

def process_symbol(symbol):
    """Process a single symbol for rounding bottom patterns and generate charts."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    local_max = argrelextrema(df["close"].values, np.greater, order=3)[0]
    local_min = argrelextrema(df["close"].values, np.less, order=3)[0]
    df["Pivot"] = 0
    for m in local_max:
        df.iloc[m, df.columns.get_loc("Pivot")] = 2
    for m in local_min:
        df.iloc[m, df.columns.get_loc("Pivot")] = 1
    df["PointPos"] = df.apply(lambda row: row['close'] - 1e-3 if row['Pivot'] == 1 else (row['close'] + 1e-3 if row['Pivot'] == 2 else np.nan), axis=1)

    all_points = find_rounding_bottom_points(df, 20)
    if all_points:
        return generate_chart(df, all_points, 20, symbol)
    return []

def load_symbols_from_csv(csv_file):
    """Load trading symbols from a CSV file."""
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Using default symbol list instead.")
        return None
    return pd.read_csv(csv_file)['Symbol'].tolist()

def fetch_symbols():
    """Fetch all linear perpetual futures pairs from Bybit."""
    exchange.load_markets()
    symbols = [
        symbol for symbol in exchange.symbols
        if symbol.endswith("/USDT:USDT")
    ]
    return symbols

def get_symbols_choice():
    """Prompt user to choose symbol input method."""
    print("\nChoose how to select symbols:")
    print("1. Scan all Bybit linear perpetual futures pairs")
    print("2. Use pairs.csv file")
    print("3. Use predefined symbol list (BTCUSDT, ETHUSDT, XRPUSDT, ADAUSDT)")
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    if choice == '1':
        return fetch_symbols(), "all Bybit linear perpetual futures pairs"
    elif choice == '2':
        symbols = load_symbols_from_csv(CSV_FILE)
        return symbols if symbols else DEFAULT_SYMBOLS, f"CSV file ({CSV_FILE})" if symbols else "default list (CSV not found)"
    else:
        return DEFAULT_SYMBOLS, "predefined symbol list"

def scan_symbols(symbols):
    """Scan provided symbols for rounding bottom patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Rounding Bottom Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"rounding_bottom_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound rounding bottom patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No rounding bottom patterns found in the scanned symbols.")