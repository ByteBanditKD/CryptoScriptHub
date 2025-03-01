#!/usr/bin/env python
"""
DoublesScanner.py
-----------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for double top and double bottom chart patterns using CCXT. It identifies local maxima and minima using scipy.signal.argrelextrema,
detects the most recent double top and double bottom patterns based on price levels within a 200-bar window, generates candlestick charts with marked patterns (only two points each),
and saves results to a CSV file. Users can choose to scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python DoublesScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/doubles_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_double_[top/bottom]_[point].png'

Source: Inspired by https://alpaca.markets/learn/algorithmic-trading-chart-pattern-python/ and 
        https://github.com/samchaaa/alpaca_tech_screener/blob/master/tech_screener_notebook.ipynb
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
INTERVAL = "1d"   # Daily interval for 200-bar window
LIMIT = 200       # Number of candles to fetch
RESULTS_FOLDER = "results"
IMG_DIR = "img"
CSV_FILE = "pairs.csv"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "AAVEUSDT", "DYDXUSDT", "IDUSDT", "JUPUSDT", "MOTHERUSDT", "OMUSDT", "ONDOUSDT", "PONKEUSDT", "STRKUSDT", "TNSRUSDT", "XLMUSDT"]
WINDOW_RANGE = 10  # Set to match your requirement for pivot detection
SMOOTH = True     # Enable smoothing for daily data
SMOOTHING_PERIOD = 3  # Reduced smoothing period for daily data
MAX_WINDOW_BARS = 200  # Maximum window for pattern detection (200 bars on daily)
PRICE_TOLERANCE = 0.05  # 10% tolerance for peak/trough price difference
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
        
        if len(df) < limit:
            print(f"Warning: Insufficient data for {adapted_symbol}. Expected {limit}, got {len(df)}")
        
        return df
    except Exception as e:
        print(f"Error fetching data for {adapted_symbol}: {e}")
        return None

def find_local_maximas_minimas(ohlc, window_range, smooth=SMOOTH, smoothing_period=SMOOTHING_PERIOD):
    """
    Find all the local maximas and minimas.

    :params ohlc: DataFrame holding the OHLC data
    :params window_range: Range to find min and max
    :params smooth: Should the prices be smoothed
    :params smoothing_period: The smoothing period

    :return max_min: DataFrame with maxima and minima
    """
    local_max_arr = []
    local_min_arr = []

    if smooth:
        smooth_close = ohlc["close"].rolling(window=smoothing_period).mean().dropna()
        local_max = argrelextrema(smooth_close.values, np.greater)[0]
        local_min = argrelextrema(smooth_close.values, np.less)[0]
    else:
        local_max = argrelextrema(ohlc["close"].values, np.greater)[0]
        local_min = argrelextrema(ohlc["close"].values, np.less)[0]

    for i in local_max:
        if (i > window_range) and (i < len(ohlc) - window_range):
            local_max_arr.append(ohlc.iloc[i - window_range:i + window_range]['close'].idxmax())

    for i in local_min:
        if (i > window_range) and (i < len(ohlc) - window_range):
            local_min_arr.append(ohlc.iloc[i - window_range:i + window_range]['close'].idxmin())

    maxima = pd.DataFrame(ohlc.loc[local_max_arr, ['close']])
    minima = pd.DataFrame(ohlc.loc[local_min_arr, ['close']])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min = max_min[~max_min.index.duplicated()]
    max_min.columns = ['Price']
    max_min['Type'] = ['Max' if idx in local_max_arr else 'Min' for idx in max_min.index]
    max_min['Index'] = range(len(max_min))  # Add integer index for pattern detection

    return max_min

def find_doubles_patterns(max_min):
    """
    Find the double tops and double bottoms patterns, returning only the most recent pattern of each type within the 200-bar window.

    :params max_min: The maximas and minimas

    :return patterns_tops, patterns_bottoms: Lists of pattern start and end indices (only latest of each type)
    """
    patterns_tops = []
    patterns_bottoms = []

    # Use a 5-candle window as per the original logic, but adjust for 200-bar window
    potential_tops = []
    potential_bottoms = []
    for i in range(5, len(max_min)):
        window = max_min.iloc[i-5:i]
        
        # Allow patterns to play out within 200 bars (days on daily data)
        if (window['Index'].iloc[-1] - window['Index'].iloc[0]) > MAX_WINDOW_BARS:
            continue
            
        a, b, c, d, e = window['Price'].values
        avg_price = (b + d) / 2  # Average price of the two peaks/troughs for tolerance

        # Double Tops (check for valid pattern)
        if (a < b and a < d and c < b and c < d and e < b and e < d and b > d and 
            abs(b - d) <= avg_price * PRICE_TOLERANCE and  # 10% tolerance
            c < (b + d) / 2 - avg_price * 0.20):  # Trough must be 20% below average peak price
            potential_tops.append((window.index[0], window.index[-1], window.index[-1]))  # Store end date for sorting

        # Double Bottoms (check for valid pattern)
        if (a > b and a > d and c > b and c > d and e > b and e > d and b < d and 
            abs(b - d) <= avg_price * PRICE_TOLERANCE and  # 10% tolerance
            c > (b + d) / 2 + avg_price * 0.20):  # Peak must be 20% above average trough price
            potential_bottoms.append((window.index[0], window.index[-1], window.index[-1]))  # Store end date for sorting

    # Sort potential tops and bottoms by end date and take the most recent of each
    if potential_tops:
        latest_top = sorted(potential_tops, key=lambda x: x[2], reverse=True)[0]
        patterns_tops.append((latest_top[0], latest_top[1]))
    if potential_bottoms:
        latest_bottom = sorted(potential_bottoms, key=lambda x: x[2], reverse=True)[0]
        patterns_bottoms.append((latest_bottom[0], latest_bottom[1]))

    return patterns_tops, patterns_bottoms

def generate_chart(ohlc, patterns_tops, patterns_bottoms, max_min, symbol):
    """Generate and save candlestick charts with only the most recent double top and double bottom patterns (two points each)."""
    results = []
    
    # Process double tops, plotting only the two peaks of the most recent pattern
    if patterns_tops:
        start_, end_ = patterns_tops[0]
        window_start = max(0, ohlc.index.get_loc(start_) - 250)  # Extended window for 200-bar context
        window_end = min(len(ohlc), ohlc.index.get_loc(end_) + 250)
        window_data = ohlc.iloc[window_start:window_end]
        if window_data.empty:
            print(f"Skipping top pattern for {symbol} at {start_}-{end_}: Empty window data")
        else:
            # Get the window of maxima/minima for the pattern
            max_min_window = max_min.loc[start_:end_]
            # Identify the two peaks (b and d) for the double top
            peaks = max_min_window[max_min_window['Type'] == 'Max']
            if len(peaks) < 2:  # Ensure exactly two peaks
                print(f"Warning: Insufficient peaks for double top in {symbol}")
            else:
                peak_dates = sorted(peaks.index)[:2]  # Take the first two peaks in chronological order

                # Create a full-length Series for plotting, filling NaN where no pattern points exist
                plot_data = pd.Series(index=window_data.index, dtype=float)
                plot_data.loc[peak_dates] = peaks.loc[peak_dates, 'Price']  # Plot only the two peak points

                addplot = [
                    mpf.make_addplot(plot_data, type='scatter', markersize=100, 
                                   marker='^', color='orange', label='Double Top Points')
                ]

                img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_double_top_{start_.strftime('%Y%m%d')}.png")
                mpf.plot(window_data, type='candle', addplot=addplot, 
                        title=f"Double Top for {symbol} at {window_data.index[-1]}",
                        style='binance', savefig=img_path, ylabel='Price (USDT)')

                results.append({
                    "Symbol": symbol,
                    "Pattern Type": "Double Top",
                    "Start Index": start_,
                    "End Index": end_,
                    "Pattern Date": window_data.index[-1],
                    "Current Price": round(window_data['close'].iloc[-1], 6),
                    "Chart Path": img_path
                })

    # Process double bottoms, plotting only the two troughs of the most recent pattern
    if patterns_bottoms:
        start_, end_ = patterns_bottoms[0]
        window_start = max(0, ohlc.index.get_loc(start_) - 250)  # Extended window for 200-bar context
        window_end = min(len(ohlc), ohlc.index.get_loc(end_) + 250)
        window_data = ohlc.iloc[window_start:window_end]
        if window_data.empty:
            print(f"Skipping bottom pattern for {symbol} at {start_}-{end_}: Empty window data")
        else:
            # Get the window of maxima/minima for the pattern
            max_min_window = max_min.loc[start_:end_]
            # Identify the two troughs (b and d) for the double bottom
            troughs = max_min_window[max_min_window['Type'] == 'Min']
            if len(troughs) < 2:  # Ensure exactly two troughs
                print(f"Warning: Insufficient troughs for double bottom in {symbol}")
            else:
                trough_dates = sorted(troughs.index)[:2]  # Take the first two troughs in chronological order

                # Create a full-length Series for plotting, filling NaN where no pattern points exist
                plot_data = pd.Series(index=window_data.index, dtype=float)
                plot_data.loc[trough_dates] = troughs.loc[trough_dates, 'Price']  # Plot only the two trough points

                addplot = [
                    mpf.make_addplot(plot_data, type='scatter', markersize=100, 
                                   marker='v', color='orange', label='Double Bottom Points')
                ]

                img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_double_bottom_{start_.strftime('%Y%m%d')}.png")
                mpf.plot(window_data, type='candle', addplot=addplot, 
                        title=f"Double Bottom for {symbol} at {window_data.index[-1]}",
                        style='binance', savefig=img_path, ylabel='Price (USDT)')

                results.append({
                    "Symbol": symbol,
                    "Pattern Type": "Double Bottom",
                    "Start Index": start_,
                    "End Index": end_,
                    "Pattern Date": window_data.index[-1],
                    "Current Price": round(window_data['close'].iloc[-1], 6),
                    "Chart Path": img_path
                })

    return results

def process_symbol(symbol):
    """Process a single symbol for double top and double bottom patterns and generate charts, focusing on the most recent pattern of each type."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    max_min = find_local_maximas_minimas(df, WINDOW_RANGE)
    patterns_tops, patterns_bottoms = find_doubles_patterns(max_min)

    if patterns_tops or patterns_bottoms:
        return generate_chart(df, patterns_tops, patterns_bottoms, max_min, symbol)
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
    """Scan provided symbols for double top and double bottom patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Doubles Pattern Scanner (Double Tops and Bottoms)!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"doubles_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound double patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No double patterns found in the scanned symbols.")