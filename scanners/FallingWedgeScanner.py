#!/usr/bin/env python
"""
FallingWedgeScanner.py
----------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for falling wedge chart patterns using CCXT. It identifies pivot points, detects falling wedges with converging downward trendlines,
generates candlestick charts with marked patterns, and saves results to a CSV file. Users can choose to scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python FallingWedgeScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/falling_wedge_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_falling_wedge_[date].png'
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
LOOKBACK = 60     # Number of periods to look back for pattern detection
PIVOT_RANGE = 5   # Number of candles for pivot detection
MIN_PIVOTS = 3    # Minimum number of pivot points for trendlines
MIN_CANDLES = 100 # Minimum candles required for analysis
SLOPE_THRESHOLD = 0.0001  # Maximum slope difference for converging lines
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
        df = df.drop_duplicates(subset='Date', keep='last')
        df.set_index('Date', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        if len(df) < MIN_CANDLES:
            print(f"Warning: Insufficient data for {adapted_symbol}. Expected at least {MIN_CANDLES}, got {len(df)}. Skipping.")
            return None
        
        return df
    except Exception as e:
        print(f"Error fetching data for {adapted_symbol}: {e}")
        return None

def pivot_id(ohlc, l, left_count, right_count):
    """Identify pivot points (highs and lows) in the OHLC data."""
    if l - left_count < 0 or l + right_count >= len(ohlc):
        return 0
    
    pivot_low = 1
    pivot_high = 1

    for i in range(max(0, l - left_count), min(len(ohlc), l + right_count + 1)):
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

def find_pivot_points(ohlc, pivot_range=PIVOT_RANGE):
    """Find all pivot highs and lows in the OHLC data."""
    ohlc["pivot"] = [pivot_id(ohlc, i, pivot_range, pivot_range) for i in range(len(ohlc))]
    return ohlc

def find_falling_wedge(ohlc, lookback=LOOKBACK, pivot_range=PIVOT_RANGE, progress=False):
    """Find falling wedge patterns in the OHLC data."""
    all_wedge_points = []
    ohlc = find_pivot_points(ohlc, pivot_range)

    if not progress:
        candle_iter = range(lookback, len(ohlc) - lookback)
    else:
        candle_iter = tqdm(range(lookback, len(ohlc) - lookback), desc="Finding falling wedge patterns...")

    for candle_idx in candle_iter:
        # Collect pivot lows (support) and highs (resistance)
        lows = np.array([])
        highs = np.array([])
        x_lows = np.array([])
        x_highs = np.array([])

        # Look back and forward for pivots
        window_start = max(0, candle_idx - lookback)
        window_end = min(len(ohlc), candle_idx + lookback)

        for i in range(window_start, window_end):
            if ohlc.iloc[i]["pivot"] == 1:  # Pivot low (support)
                lows = np.append(lows, ohlc.iloc[i]["low"])
                x_lows = np.append(x_lows, i)
            elif ohlc.iloc[i]["pivot"] == 2:  # Pivot high (resistance)
                highs = np.append(highs, ohlc.iloc[i]["high"])
                x_highs = np.append(x_highs, i)

        if len(x_lows) < MIN_PIVOTS or len(x_highs) < MIN_PIVOTS:
            continue

        # Fit trendlines for support (lower line) and resistance (upper line)
        try:
            slope_low, intercept_low, _, _, _ = linregress(x_lows, lows)
            slope_high, intercept_high, _, _, _ = linregress(x_highs, highs)

            # Ensure both lines are downward sloping (negative slopes)
            if slope_low >= 0 or slope_high >= 0:
                continue

            # Ensure lines are converging (difference in slopes is small and positive)
            slope_diff = slope_high - slope_low
            if slope_diff > SLOPE_THRESHOLD or slope_diff < -SLOPE_THRESHOLD:
                continue

            # Ensure the pattern is tightening (resistance closer to support over time)
            x_range = np.linspace(window_start, window_end, 100)
            y_low = slope_low * x_range + intercept_low
            y_high = slope_high * x_range + intercept_high
            if not (np.all(np.diff(y_high - y_low) < 0)):  # Ensure convergence
                continue

            # Check breakout (price breaks above resistance)
            breakout_idx = candle_idx
            if breakout_idx < window_end and ohlc.iloc[breakout_idx]["close"] > y_high[-1]:
                all_wedge_points.append(breakout_idx)

        except ValueError:
            continue

    return all_wedge_points, (slope_low, intercept_low, slope_high, intercept_high)

def generate_chart(ohlc, wedge_points, trendlines, symbol):
    """Generate and save candlestick chart with falling wedge pattern."""
    results = []
    for point in tqdm(wedge_points, desc=f"Generating falling wedge charts for {symbol}", leave=False):
        lookback = LOOKBACK
        start_idx = max(0, point - lookback - 10)
        end_idx = min(len(ohlc), point + lookback + 20)  # Extended for target and pullback
        window_data = ohlc.iloc[start_idx:end_idx]
        if window_data.empty:
            print(f"Skipping falling wedge pattern for {symbol} at {point}: Empty window data")
            continue

        slope_low, intercept_low, slope_high, intercept_high = trendlines
        x_range = np.linspace(start_idx, end_idx, len(window_data))
        lower_line = slope_low * x_range + intercept_low
        upper_line = slope_high * x_range + intercept_high

        # Calculate target (height of wedge added to breakout)
        wedge_height = max(upper_line[0] - lower_line[0], upper_line[-1] - lower_line[-1])
        breakout_price = ohlc.iloc[point]["close"]
        target_price = breakout_price + wedge_height

        # Add pullback and target lines (numeric y values only)
        pullback_level = (upper_line[-1] + lower_line[-1]) / 2  # Midpoint as pullback
        hlines = [round(pullback_level, 6), round(target_price, 6)]  # Simplified to list of y values

        addplot = [
            mpf.make_addplot(upper_line, type='line', color='blue', label='Resistance'),
            mpf.make_addplot(lower_line, type='line', color='purple', label='Support')  # Changed "Support" to label="Support"
        ]

        img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_falling_wedge_{window_data.index[-1].strftime('%Y%m%d_%H%M%S')}.png")
        mpf.plot(window_data, type='candle', addplot=addplot, hlines=hlines,
                title=f"Falling Wedge for {symbol} at {window_data.index[-1]}",
                style='binance', savefig=img_path, ylabel='Price (USDT)')

        results.append({
            "Symbol": symbol,
            "Pattern Index": point,
            "Pattern Date": window_data.index[-1],
            "Breakout Price": round(breakout_price, 6),
            "Target Price": round(target_price, 6),
            "Pullback Level": round(pullback_level, 6),
            "Chart Path": img_path
        })

    return results

def process_symbol(symbol):
    """Process a single symbol for falling wedge patterns and generate charts."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    wedge_points, trendlines = find_falling_wedge(df, lookback=LOOKBACK, pivot_range=PIVOT_RANGE, progress=True)
    if wedge_points:
        return generate_chart(df, wedge_points, trendlines, symbol)
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
    """Scan provided symbols for falling wedge patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Falling Wedge Pattern Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"falling_wedge_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound falling wedge patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No falling wedge patterns found in the scanned symbols.")