#!/usr/bin/env python
"""
WedgePatternScanner.py
----------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for wedge patterns using CCXT. It identifies pivot points, detects wedge formations
using linear regression, generates candlestick charts with marked patterns, and saves results to a CSV file. Users can choose to
scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python WedgePatternScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/wedge_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_wedge_[point].png'
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
LIMIT = 200       # Number of candles to fetch
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
        df.set_index('Date', inplace=True, drop=False)
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
        if ohlc.loc[ohlc.index[l], "close"] > ohlc.loc[ohlc.index[i], "close"]:
            pivot_low = 0
        if ohlc.loc[ohlc.index[l], "close"] < ohlc.loc[ohlc.index[i], "close"]:
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
        return row['close'] - 1e-3
    elif row['Pivot'] == 2:
        return row['close'] + 1e-3
    else:
        return np.nan

def find_wedge_points(ohlc, back_candles):
    """Detect wedge patterns using linear regression on pivot points."""
    all_points = []
    for candle_idx in range(back_candles + 10, len(ohlc)):
        maxim = np.array([])
        minim = np.array([])
        xxmin = np.array([])
        xxmax = np.array([])
        for i in range(candle_idx - back_candles, candle_idx + 1):
            if ohlc.loc[ohlc.index[i], "Pivot"] == 1:
                minim = np.append(minim, ohlc.loc[ohlc.index[i], "close"])
                xxmin = np.append(xxmin, i)
            if ohlc.loc[ohlc.index[i], "Pivot"] == 2:
                maxim = np.append(maxim, ohlc.loc[ohlc.index[i], "close"])
                xxmax = np.append(xxmax, i)
        
        # Ensure at least 2 unique points for regression
        if len(np.unique(xxmax)) < 2 or len(np.unique(xxmin)) < 2:
            continue
        
        # Perform linear regression with error handling
        try:
            slmin, intercmin, rmin, _, _ = linregress(xxmin, minim)
            slmax, intercmax, rmax, _, _ = linregress(xxmax, maxim)
        except ValueError as e:
            print(f"Regression error for {ohlc.index[candle_idx]}: {e}")
            continue
        
        # Check for valid wedge pattern
        if (np.isnan(slmin) or np.isnan(slmax) or np.isnan(rmin) or np.isnan(rmax)):
            continue
        
        if abs(rmax) >= 0.9 and abs(rmin) >= 0.9 and ((slmin >= 1e-3 and slmax >= 1e-3) or (slmin <= -1e-3 and slmax <= -1e-3)):
            x_ = (intercmin - intercmax) / (slmax - slmin)
            cors = np.hstack([xxmax, xxmin])
            if (x_ - max(cors)) > 0 and (x_ - max(cors)) < (max(cors) - min(cors)) * 3:
                all_points.append(candle_idx)
    return all_points

def generate_chart(ohlc, all_points, back_candles, symbol):
    """Generate and save candlestick chart with wedge pattern markers."""
    results = []
    for point in all_points:
        start_idx = max(0, point - back_candles)
        window_data = ohlc.iloc[start_idx:point + 1]
        
        addplot = [
            mpf.make_addplot(window_data['PointPos'], type='scatter', markersize=100, 
                           marker='^' if window_data['Pivot'].iloc[-1] == 2 else 'v', 
                           color='blue', label='Pivot Points')
        ]
        
        img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_wedge_{point}.png")
        mpf.plot(window_data, type='candle', addplot=addplot, 
                title=f"Wedge Pattern for {symbol} at {window_data.index[-1]}",
                style='binance', savefig=img_path, ylabel='Price (USDT)')
        
        results.append({
            "Symbol": symbol,
            "Wedge Point Index": point,
            "Wedge Point Date": window_data.index[-1],
            "Current Price": round(window_data['close'].iloc[-1], 6),
            "Chart Path": img_path
        })
    return results

def process_symbol(symbol):
    """Process a single symbol for wedge patterns and generate charts."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    df["Pivot"] = df.apply(lambda x: pivot_id(df, df.index.get_loc(x.name), 3, 3), axis=1)
    df["PointPos"] = df.apply(lambda row: pivot_point_position(row), axis=1)
    all_points = find_wedge_points(df, 20)
    
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
    """Scan provided symbols for wedge patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Wedge Pattern Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"wedge_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound wedge patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No wedge patterns found in the scanned symbols.")