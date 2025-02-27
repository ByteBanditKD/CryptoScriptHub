"""
TD_Sequential_Scanner.py
------------------------
This script scans cryptocurrency trading pairs on Bybit for TD Sequential signals (TD9 and TD13)
using the CCXT library. Users can specify which timeframes to check for signals.

Requirements:
- pip install ccxt
- Python 3.7+

Usage:
- Edit the SELECTED_TIMEFRAMES list below to choose which timeframes to scan
- Available options: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
- Run directly: python TD_Sequential_Scanner.py
- Results are saved in 'results/td_sequential_signals.csv'
"""

import ccxt
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# User Configuration - Choose timeframes to scan
SELECTED_TIMEFRAMES = ['1w']  # Edit this list to select timeframes

# Constants
EXCHANGE = ccxt.bybit()
MAX_WORKERS = 1      # Number of concurrent threads
OUTPUT_DIR = "results"
OUTPUT_CSV = f"{OUTPUT_DIR}/td_sequential_signals.csv"
AVAILABLE_TIMEFRAMES = {
    "1m": "1",    # 1 minute
    "5m": "5",    # 5 minutes
    "15m": "15",  # 15 minutes
    "30m": "30",  # 30 minutes
    "1h": "60",   # 1 hour
    "4h": "240",  # 4 hours
    "1d": "D",    # 1 day
    "1w": "W"     # 1 week
}

def validate_timeframes():
    """Validate selected timeframes against available options."""
    invalid = [tf for tf in SELECTED_TIMEFRAMES if tf not in AVAILABLE_TIMEFRAMES]
    if invalid:
        raise ValueError(f"Invalid timeframes selected: {invalid}. Available options: {list(AVAILABLE_TIMEFRAMES.keys())}")
    return {tf: AVAILABLE_TIMEFRAMES[tf] for tf in SELECTED_TIMEFRAMES}

def ensure_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_trading_pairs():
    """Fetch available USDT trading pairs from Bybit."""
    try:
        EXCHANGE.load_markets()
        return [symbol for symbol in EXCHANGE.markets.keys() if symbol.endswith('USDT')]
    except Exception as e:
        print(f"Error fetching trading pairs: {e}")
        return []

def fetch_ohlcv(symbol, timeframe):
    """Fetch OHLCV data for a given symbol and timeframe."""
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=100)
        return [[str(c[0]), str(c[1]), str(c[2]), str(c[3]), str(c[4]), str(c[5])] for c in ohlcv]
    except Exception as e:
        print(f"Failed to fetch OHLCV for {symbol} on {timeframe}: {e}")
        return []

def detect_td_sequential(candles, check_td9=False, check_td13=False):
    """Detect TD9 and TD13 sequential patterns in candle data."""
    td9 = False
    td13 = False

    if not candles or len(candles) < 14:
        return td9, td13

    if check_td9 and len(candles) >= 9:
        for i in range(9, len(candles)):
            if all(float(candles[i - j][4]) > float(candles[i - j - 4][4]) for j in range(9)):
                td9 = True
                break

    if check_td13 and len(candles) >= 14:
        for i in range(13, len(candles)):
            if all(float(candles[i - j][4]) < float(candles[i - j - 1][4]) for j in range(13)):
                td13 = True
                break

    return td9, td13

def process_pair_interval(pair, timeframe, check_td9, check_td13):
    """Process a single trading pair and timeframe for TD signals."""
    candles = fetch_ohlcv(pair, timeframe)
    if candles:
        td9, td13 = detect_td_sequential(candles, check_td9, check_td13)
        if td9 or td13:
            signal_type = "TD9" if td9 else "TD13"
            print(f"Signal detected: {pair}, {timeframe}, {signal_type}")
            return [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pair, timeframe, td9, td13]
    return None

def write_to_csv(results):
    """Write results to CSV file."""
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Symbol", "Timeframe", "TD9", "TD13"])
        for result in results:
            writer.writerow(result)

def scanner():
    """Run a single scan for TD signals."""
    # Validate selected timeframes
    try:
        timeframes = validate_timeframes()
    except ValueError as e:
        print(e)
        return

    ensure_results_directory()
    pairs = get_trading_pairs()
    if not pairs:
        print("No trading pairs available. Exiting.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        # Schedule tasks for selected timeframes
        for pair in pairs:
            for tf in timeframes.keys():
                # Check both TD9 and TD13 for each timeframe
                futures.append(executor.submit(process_pair_interval, pair, tf, True, True))

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    if results:
        write_to_csv(results)
        print(f"Scan complete. Wrote {len(results)} signals to {OUTPUT_CSV}")
    else:
        print("Scan complete. No signals detected.")

if __name__ == "__main__":
    try:
        scanner()
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")