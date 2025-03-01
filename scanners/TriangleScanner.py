#!/usr/bin/env python
"""
TriangleScanner.py
------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for triangle chart patterns (symmetrical, ascending, descending) using CCXT. It identifies pivot points,
detects triangle formations with linear regression, generates candlestick charts with marked patterns, and saves results to a CSV file. Users can choose
to scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python TriangleScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/triangle_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_triangle_[point].png'

Source: Inspired by https://www.youtube.com/watch?v=WVNB_6JRbl0
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime
import mplfinance as mpf
from scipy.stats import linregress
from tqdm import tqdm

# ================ CONFIGURATIONS ================= #
EXCHANGE_NAME = "bybit"
INTERVAL = "240"  # 4-hour interval
LIMIT = 50       # Number of candles to fetch
RESULTS_FOLDER = "results"
IMG_DIR = "img"
CSV_FILE = "pairs.csv"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"]
TRIANGLE_TYPE = "symmetrical"  # Options: 'symmetrical', 'ascending', 'descending'
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

def pivot_id(ohlc, l, n1, n2):
    """Identify pivot points (highs and lows) in the OHLC data."""
    if l - n1 < 0 or l + n2 >= len(ohlc):
        return 0
    
    pivot_low = 1
    pivot_high = 1

    for i in range(l - n1, l + n2 + 1):
        if ohlc.iloc[l]["low"] > ohlc.iloc[i]["low"]:
            pivot_low = 0
        if ohlc.iloc[l]["high"] < ohlc.iloc[i]["high"]:
            pivot_high = 0

    if pivot_low and pivot_high:
        return 3
    elif pivot_low:
        return 1
    elif pivot_high:
        return 2
    else:
        return 0

def pivot_point_position(row):
    """Assign position for plotting pivot points."""
    if row['Pivot'] == 1:
        return row['low'] - 1e-3
    elif row['Pivot'] == 2:
        return row['high'] + 1e-3
    else:
        return np.nan

def find_triangle_points(ohlc, back_candles, triangle_type=TRIANGLE_TYPE):
    """Find triangle points based on pivot points."""
    all_triangle_points = []
    for candle_idx in range(back_candles + 10, len(ohlc)):
        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])

        for i in range(candle_idx - back_candles, candle_idx + 1):
            if ohlc.iloc[i]["Pivot"] == 1:
                minim = np.append(minim, ohlc.iloc[i]["low"])
                xxmin = np.append(xxmin, i)
            if ohlc.iloc[i]["Pivot"] == 2:
                maxim = np.append(maxim, ohlc.iloc[i]["high"])
                xxmax = np.append(xxmax, i)

        # Ensure at least 2 unique points for regression
        if len(np.unique(xxmax)) < 2 or len(np.unique(xxmin)) < 2:
            continue

        try:
            slmin, intercmin, rmin, _, _ = linregress(xxmin, minim)
            slmax, intercmax, rmax, _, _ = linregress(xxmax, maxim)
        except ValueError as e:
            print(f"Regression error for {ohlc.index[candle_idx]} in {triangle_type} triangle: {e}")
            continue

        # Check for NaN values in regression results
        if np.isnan(slmin) or np.isnan(slmax) or np.isnan(rmin) or np.isnan(rmax):
            continue

        if triangle_type == "symmetrical":
            if abs(rmax) >= 0.9 and abs(rmin) >= 0.9 and slmin >= 0.0001 and slmax <= -0.0001:
                all_triangle_points.append(candle_idx)
        elif triangle_type == "ascending":
            if abs(rmax) >= 0.9 and abs(rmin) >= 0.9 and slmin >= 0.0001 and -0.00001 <= slmax <= 0.00001:
                all_triangle_points.append(candle_idx)
        elif triangle_type == "descending":
            if abs(rmax) >= 0.9 and abs(rmin) >= 0.9 and slmax <= -0.0001 and -0.00001 <= slmin <= 0.00001:
                all_triangle_points.append(candle_idx)

    return all_triangle_points

def generate_chart(ohlc, all_triangle_points, back_candles, symbol):
    """Generate and save candlestick chart with triangle pattern."""
    results = []
    for point in all_triangle_points:
        start_idx = max(0, point - back_candles)
        if start_idx >= len(ohlc) or point + 1 > len(ohlc):
            print(f"Skipping point {point} for {symbol}: Index out of bounds (start_idx: {start_idx}, point: {point}, len: {len(ohlc)})")
            continue
        
        window_data = ohlc.iloc[start_idx:point + 1]
        if window_data.empty:
            print(f"Skipping point {point} for {symbol}: Empty window data")
            continue
        
        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])
        for i in range(start_idx, point + 1):
            if ohlc.iloc[i]["Pivot"] == 1:
                minim = np.append(minim, ohlc.iloc[i]["low"])
                xxmin = np.append(xxmin, i - start_idx)
            if ohlc.iloc[i]["Pivot"] == 2:
                maxim = np.append(maxim, ohlc.iloc[i]["high"])
                xxmax = np.append(xxmax, i - start_idx)

        if len(xxmin) < 2 or len(xxmax) < 2:
            continue

        slmin, intercmin, _, _, _ = linregress(xxmin, minim)
        slmax, intercmax, _, _, _ = linregress(xxmax, maxim)
        x_extended = np.linspace(0, len(window_data) + 15, len(window_data) + 15)
        lower_line = slmin * x_extended + intercmin
        upper_line = slmax * x_extended + intercmax

        addplot = [
            mpf.make_addplot(window_data['PointPos'], type='scatter', markersize=100, 
                           marker='v' if window_data['Pivot'].iloc[0] == 1 else '^', 
                           color='blue', label='Pivot Points'),
            mpf.make_addplot(lower_line[:len(window_data)], type='line', color='purple', label='Lower Trendline'),
            mpf.make_addplot(upper_line[:len(window_data)], type='line', color='purple', label='Upper Trendline')
        ]

        img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_triangle_{point}.png")
        mpf.plot(window_data, type='candle', addplot=addplot, 
                title=f"Triangle Pattern for {symbol} at {window_data.index[-1]}",
                style='binance', savefig=img_path, ylabel='Price (USDT)')

        results.append({
            "Symbol": symbol,
            "Triangle Point Index": point,
            "Triangle Point Date": window_data.index[-1],
            "Current Price": round(window_data['close'].iloc[-1], 6),
            "Chart Path": img_path
        })
    return results

def process_symbol(symbol):
    """Process a single symbol for triangle patterns and generate charts."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    df["Pivot"] = [pivot_id(df, i, 3, 3) for i in range(len(df))]
    df["PointPos"] = df.apply(pivot_point_position, axis=1)

    all_triangle_points = find_triangle_points(df, 20, TRIANGLE_TYPE)
    if all_triangle_points:
        return generate_chart(df, all_triangle_points, 20, symbol)
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
    """Scan provided symbols for triangle patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Triangle Pattern Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"triangle_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound triangle patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No triangle patterns found in the scanned symbols.")