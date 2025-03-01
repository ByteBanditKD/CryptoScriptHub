#!/usr/bin/env python
"""
HeadAndShouldersScanner.py
--------------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for head and shoulders and inverse head and shoulders chart patterns using CCXT. It identifies pivot points,
detects patterns with linear regression and price thresholds, generates candlestick charts with marked patterns, and saves results to a CSV file. Users can choose
to scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm ta-lib
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python HeadAndShouldersScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/hs_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_[hs/inverse-hs]_[date].png'

Source: Inspired by https://www.youtube.com/watch?v=Mxk8PP3vbuA
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime
import mplfinance as mpf
from scipy.stats import linregress
from tqdm import tqdm
import talib  # For ATR calculation

# ================ CONFIGURATIONS ================= #
EXCHANGE_NAME = "bybit"
INTERVAL = "60"  # 4-hour interval
LIMIT = 200       # Number of candles to fetch
RESULTS_FOLDER = "results"
IMG_DIR = "img"
CSV_FILE = "pairs.csv"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"]
BACK_CANDLES = 40  # Increased look-back and look-forward period
MIN_CANDLES = 50   # Minimum candles required for analysis
PIVOT_RANGE_SHORT = 7  # Increased pivot range for ShortPivot
PIVOT_RANGE_LONG = 10  # Increased pivot range for Pivot
PRICE_THRESHOLD_FACTOR = 0.05  # 5% of price range as threshold
NECKLINE_SLOPE_THRESHOLD = 2e-4  # Tighter neckline slope
NECKLINE_CORRELATION_THRESHOLD = 0.9  # Stronger correlation for neckline
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

def pivot_id(ohlc, l, n1, n2):
    """Identify pivot points (highs and lows) in the OHLC data with dynamic threshold."""
    if l - n1 < 0 or l + n2 >= len(ohlc):
        return 0
    
    pivot_low = 1
    pivot_high = 1

    # Dynamic threshold based on ATR or price range
    atr = talib.ATR(ohlc['high'].values, ohlc['low'].values, ohlc['close'].values, timeperiod=BACK_CANDLES)[-1]
    if np.isnan(atr) or atr == 0:
        atr = ohlc.iloc[l]["close"] * 0.01  # Default to 1% of price if ATR fails
    price_threshold = max(atr, ohlc.iloc[l]["close"] * PRICE_THRESHOLD_FACTOR)  # Use max of ATR or 5% price

    for i in range(max(0, l - n1), min(len(ohlc), l + n2 + 1)):
        if ohlc.iloc[l]["low"] > ohlc.iloc[i]["low"] + price_threshold:
            pivot_low = 0
        if ohlc.iloc[l]["high"] < ohlc.iloc[i]["high"] - price_threshold:
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

def _find_points(df, candle_id, back_candles):
    """Find points for head and shoulders pattern detection."""
    maxim = np.array([])
    minim = np.array([])
    xxmax = np.array([])
    xxmin = np.array([])
    minbcount = 0  # minimas before head
    maxbcount = 0  # maximas before head
    minacount = 0  # minimas after head
    maxacount = 0  # maximas after head
    
    window_start = max(0, candle_id - back_candles)
    window_end = min(len(df), candle_id + back_candles)
    
    for i in range(window_start, window_end):
        if df.iloc[i]["ShortPivot"] == 1:
            minim = np.append(minim, df.iloc[i]["low"])
            xxmin = np.append(xxmin, i)
            if i < candle_id:
                minbcount += 1
            elif i > candle_id:
                minacount += 1
        if df.iloc[i]["ShortPivot"] == 2:
            maxim = np.append(maxim, df.iloc[i]["high"])
            xxmax = np.append(xxmax, i)
            if i < candle_id:
                maxbcount += 1
            elif i > candle_id:
                maxacount += 1

    return maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount

def find_inverse_head_and_shoulders(df, back_candles=BACK_CANDLES):
    """Find all inverse head and shoulders chart patterns."""
    all_points = []
    for candle_id in range(back_candles + 20, len(df) - back_candles):
        if df.iloc[candle_id]["Pivot"] != 1 or df.iloc[candle_id]["ShortPivot"] != 1:
            continue
        
        maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount = _find_points(df, candle_id, back_candles)
        if minbcount < 1 or minacount < 1 or maxbcount < 1 or maxacount < 1:
            continue

        try:
            slmax, intercmax, rmax, _, _ = linregress(xxmax, maxim)
            headidx = np.argmin(minim, axis=0)
            price = df.iloc[candle_id]["close"]
            # Ensure shoulders are within 2% of each other and head is 5% lower
            if (headidx > 0 and headidx < len(minim) - 1 and
                minim[headidx] < minim[headidx-1] * (1 - 0.05) and  # 5% lower
                minim[headidx] < minim[headidx+1] * (1 - 0.05) and
                abs(minim[headidx-1] - minim[headidx+1]) <= minim[headidx-1] * 0.02 and  # Shoulders within 2%
                abs(slmax) <= NECKLINE_SLOPE_THRESHOLD and
                abs(rmax) >= NECKLINE_CORRELATION_THRESHOLD):
                all_points.append(candle_id)
        except (ValueError, IndexError):
            continue

    return all_points

def find_head_and_shoulders(df, back_candles=BACK_CANDLES):
    """Find all head and shoulders chart patterns."""
    all_points = []
    for candle_id in range(back_candles + 20, len(df) - back_candles):
        if df.iloc[candle_id]["Pivot"] != 2 or df.iloc[candle_id]["ShortPivot"] != 2:
            continue

        maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount = _find_points(df, candle_id, back_candles)
        if minbcount < 1 or minacount < 1 or maxbcount < 1 or maxacount < 1:
            continue

        try:
            slmin, intercmin, rmin, _, _ = linregress(xxmin, minim)
            headidx = np.argmax(maxim, axis=0)
            price = df.iloc[candle_id]["close"]
            if (len(maxim) > 2 and 0 < headidx < len(maxim) - 1 and
                maxim[headidx] > maxim[headidx - 1] * (1 + 0.05) and  # 5% higher
                maxim[headidx] > maxim[headidx + 1] * (1 + 0.05) and
                abs(maxim[headidx-1] - maxim[headidx+1]) <= maxim[headidx-1] * 0.02 and  # Shoulders within 2%
                abs(slmin) <= NECKLINE_SLOPE_THRESHOLD and
                abs(rmin) >= NECKLINE_CORRELATION_THRESHOLD):
                all_points.append(candle_id)
        except (ValueError, IndexError):
            continue

    return all_points

def generate_chart(ohlc, all_hs_points, all_ihs_points, back_candles, symbol):
    """Generate and save candlestick charts with head and shoulders patterns."""
    results = []

    for pattern_type, all_points in [("hs", all_hs_points), ("inverse-hs", all_ihs_points)]:
        for point in tqdm(all_points, desc=f"Generating {pattern_type} charts for {symbol}", leave=False):
            start_idx = max(0, point - (back_candles + 6))
            end_idx = min(len(ohlc), point + back_candles + 40)  # Extended window for context
            window_data = ohlc.iloc[start_idx:end_idx]
            if window_data.empty:
                print(f"Skipping {pattern_type} pattern for {symbol} at {point}: Empty window data")
                continue

            maxim = np.array([])
            minim = np.array([])
            xxmin = np.array([])
            xxmax = np.array([])
            for i in range(start_idx, end_idx):
                if ohlc.iloc[i]["ShortPivot"] == 1:
                    minim = np.append(minim, ohlc.iloc[i]["low"])
                    xxmin = np.append(xxmin, i)
                if ohlc.iloc[i]["ShortPivot"] == 2:
                    maxim = np.append(maxim, ohlc.iloc[i]["high"])
                    xxmax = np.append(xxmax, i)

            if pattern_type == "hs":
                headidx = np.argmax(maxim, axis=0)
                try:
                    hsx = [xxmax[headidx-1], xxmin[0], xxmax[headidx], xxmin[1], xxmax[headidx+1]]
                    hsy = [maxim[headidx-1], minim[0], maxim[headidx], minim[1], maxim[headidx+1]]
                except IndexError:
                    continue
            else:
                headidx = np.argmin(minim, axis=0)
                try:
                    hsx = [xxmin[headidx-1], xxmax[0], xxmin[headidx], xxmax[1], xxmin[headidx+1]]
                    hsy = [minim[headidx-1], maxim[0], minim[headidx], maxim[1], minim[headidx+1]]
                except IndexError:
                    continue

            # Calculate neckline (least squares fit of lows or highs)
            if pattern_type == "hs":
                neckline_points = minim[xxmin < xxmax[headidx]]
                neckline_x = xxmin[xxmin < xxmax[headidx]]
            else:
                neckline_points = maxim[xxmax < xxmin[headidx]]
                neckline_x = xxmax[xxmax < xxmin[headidx]]

            if len(neckline_x) >= 3:  # Require at least 3 points for neckline
                try:
                    neckline_slope, neckline_intercept, neckline_r, _, _ = linregress(neckline_x, neckline_points)
                    if (abs(neckline_slope) > NECKLINE_SLOPE_THRESHOLD or
                        abs(neckline_r) < NECKLINE_CORRELATION_THRESHOLD):
                        continue  # Skip if neckline slope is too steep or correlation is weak
                except ValueError:
                    continue
            else:
                continue

            levels = [(ohlc.index[int(x)], y) for x, y in zip(hsx, hsy)]
            hs_points = pd.Series(index=window_data.index, dtype=float)
            for x, y in levels:
                hs_points.loc[x] = y

            addplot = [
                mpf.make_addplot(hs_points, type='scatter', markersize=200, marker='v', color='r', label='Pattern Points')
            ]

            img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_{pattern_type}_{window_data.index[-1].strftime('%Y%m%d_%H%M%S')}.png")
            mpf.plot(window_data, type='candle', addplot=addplot, alines=dict(alines=levels, colors=['purple'], alpha=0.5, linewidths=20),
                    title=f"{pattern_type.capitalize()} for {symbol} at {window_data.index[-1]}",
                    style='binance', savefig=img_path, ylabel='Price (USDT)')

            results.append({
                "Symbol": symbol,
                "Pattern Type": "Head and Shoulders" if pattern_type == "hs" else "Inverse Head and Shoulders",
                "Pattern Index": point,
                "Pattern Date": window_data.index[-1],
                "Current Price": round(window_data['close'].iloc[-1], 6),
                "Chart Path": img_path
            })

    return results

def process_symbol(symbol):
    """Process a single symbol for head and shoulders patterns and generate charts."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    # Optimize pivot point calculation with tqdm for progress feedback
    tqdm.pandas(desc=f"Computing ShortPivot for {symbol}")
    df["ShortPivot"] = df.index.map(lambda i: pivot_id(df, df.index.get_loc(i), 3, PIVOT_RANGE_SHORT))
    tqdm.pandas(desc=f"Computing Pivot for {symbol}")
    df["Pivot"] = df.index.map(lambda i: pivot_id(df, df.index.get_loc(i), PIVOT_RANGE_LONG, PIVOT_RANGE_LONG))
    df["PointPos"] = df.apply(pivot_point_position, axis=1)

    all_hs_points = find_head_and_shoulders(df)
    all_ihs_points = find_inverse_head_and_shoulders(df)

    if all_hs_points or all_ihs_points:
        return generate_chart(df, all_hs_points, all_ihs_points, BACK_CANDLES, symbol)
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
    """Scan provided symbols for head and shoulders patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Head and Shoulders Pattern Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"hs_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No head and shoulders patterns found in the scanned symbols.")