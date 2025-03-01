"""
InverseHeadAndShouldersScanner.py
---------------------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit linear perpetual futures for inverse head and shoulders chart patterns using CCXT. It identifies pivot points,
detects patterns with the specified logic, generates candlestick charts with marked patterns, and saves results to a CSV file. Users can choose
to scan all futures pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy matplotlib mplfinance scipy tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python InverseHeadAndShouldersScanner.py
- Follow prompts to choose input method (all futures, CSV, or symbol list)
- Results are saved in 'results/ihs_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_inverse-hs_[date].png'

Source: Inspired by Zetra Team logic from 2023-01-09
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
INTERVAL = "1h"  # 4-hour interval
LIMIT = 200       # Number of candles to fetch
RESULTS_FOLDER = "results"
IMG_DIR = "img"
CSV_FILE = "pairs.csv"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"]
LOOKBACK = 60     # Number of periods to use for back candles
PIVOT_INTERVAL = 10  # Number of candles for pivot detection
SHORT_PIVOT_INTERVAL = 5  # Shorter pivot interval (must be < PIVOT_INTERVAL)
HEAD_RATIO_BEFORE = 0.98  # Ratio between head and left shoulder
HEAD_RATIO_AFTER = 0.98   # Ratio between head and right shoulder
UPPER_SLMAX = 1e-4       # Upper limit of neckline slope
MIN_CANDLES = 100        # Minimum candles required for analysis
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

def pivot_id(ohlc, l, left_count, right_count, name_pivot="pivot"):
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

def find_points(ohlc, candle_idx, lookback):
    """Find pivot points and counts for pattern detection."""
    maxim = np.array([])
    minim = np.array([])
    xxmax = np.array([])
    xxmin = np.array([])
    minbcount = 0  # minimas before head
    maxbcount = 0  # maximas before head
    minacount = 0  # minimas after head
    maxacount = 0  # maximas after head
    
    window_start = max(0, candle_idx - lookback)
    window_end = min(len(ohlc), candle_idx + lookback)
    
    for i in range(window_start, window_end):
        if ohlc.iloc[i]["short_pivot"] == 1:
            minim = np.append(minim, ohlc.iloc[i]["low"])
            xxmin = np.append(xxmin, i)
            if i < candle_idx:
                minbcount += 1
            elif i > candle_idx:
                minacount += 1
        if ohlc.iloc[i]["short_pivot"] == 2:
            maxim = np.append(maxim, ohlc.iloc[i]["high"])
            xxmax = np.append(xxmax, i)
            if i < candle_idx:
                maxbcount += 1
            elif i > candle_idx:
                maxacount += 1

    return maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount

def find_inverse_head_and_shoulders(ohlc, lookback=LOOKBACK, pivot_interval=PIVOT_INTERVAL, 
                                   short_pivot_interval=SHORT_PIVOT_INTERVAL, head_ratio_before=HEAD_RATIO_BEFORE, 
                                   head_ratio_after=HEAD_RATIO_AFTER, upper_slmax=UPPER_SLMAX, progress=False):
    """Find all inverse head and shoulders chart patterns."""
    if short_pivot_interval <= 0 or pivot_interval <= 0:
        raise ValueError("Value cannot be less than or equal to 0")
    if short_pivot_interval >= pivot_interval:
        raise ValueError(f"short_pivot_interval must be less than pivot_interval")

    # Initialize columns
    ohlc["ihs_lookback"] = lookback
    ohlc["chart_type"] = ""
    ohlc["ihs_idx"] = [np.array([]) for _ in range(len(ohlc))]
    ohlc["ihs_point"] = [np.array([]) for _ in range(len(ohlc))]

    # Find pivot points
    ohlc["pivot"] = [pivot_id(ohlc, i, pivot_interval, pivot_interval) for i in range(len(ohlc))]
    ohlc["short_pivot"] = [pivot_id(ohlc, i, short_pivot_interval, short_pivot_interval, "short_pivot") for i in range(len(ohlc))]

    if not progress:
        candle_iter = range(lookback, len(ohlc))
    else:
        candle_iter = tqdm(range(lookback, len(ohlc)), desc="Finding inverse head and shoulder patterns...")

    for candle_idx in candle_iter:
        if ohlc.iloc[candle_idx]["pivot"] != 1 or ohlc.iloc[candle_idx]["short_pivot"] != 1:
            continue
        
        maxim, minim, xxmax, xxmin, maxacount, minacount, maxbcount, minbcount = find_points(ohlc, candle_idx, lookback)
        if minbcount < 1 or minacount < 1 or maxbcount < 1 or maxacount < 1:
            continue

        try:
            slmax, _, _, _, _ = linregress(xxmax, maxim)
            headidx = np.argmin(minim, axis=0)
            if (len(minim) - 1 == headidx):  # Skip if head is the last value
                continue
            
            if (minim[headidx-1] - minim[headidx] > 0 and
                minim[headidx] / minim[headidx-1] < 1 and minim[headidx] / minim[headidx-1] >= head_ratio_before and
                minim[headidx] / minim[headidx+1] < 1 and minim[headidx] / minim[headidx+1] >= head_ratio_after and
                minim[headidx+1] - minim[headidx] > 0 and
                abs(slmax) <= upper_slmax and
                xxmax[0] > xxmin[headidx-1] and xxmax[1] < xxmin[headidx+1]):
                
                ohlc.loc[candle_idx, "chart_type"] = "inverse-hs"
                indexes = [int(xxmin[headidx-1]), int(xxmax[0]), int(xxmin[headidx]), int(xxmax[1]), int(xxmin[headidx+1])]
                values = [minim[headidx-1], maxim[0], minim[headidx], maxim[1], minim[headidx+1]]
                list_idx_values = [(i, v) for i, v in zip(indexes, values)]
                ohlc.at[candle_idx, "ihs_idx"] = [t[0] for t in list_idx_values]
                ohlc.at[candle_idx, "ihs_point"] = [t[1] for t in list_idx_values]
        except (ValueError, IndexError):
            continue

    return ohlc

def generate_chart(ohlc, symbol):
    """Generate and save candlestick chart with inverse head and shoulders pattern."""
    results = []
    inverse_hs_points = ohlc[ohlc["chart_type"] == "inverse-hs"].index
    
    for point in tqdm(inverse_hs_points, desc=f"Generating inverse-hs charts for {symbol}", leave=False):
        lookback = ohlc.loc[point, "ihs_lookback"]
        start_idx = max(0, point - lookback - 6)
        end_idx = min(len(ohlc), point + lookback + 6)
        window_data = ohlc.iloc[start_idx:end_idx]
        if window_data.empty:
            print(f"Skipping inverse-hs pattern for {symbol} at {point}: Empty window data")
            continue

        # Extract pattern points
        ihs_idx = ohlc.loc[point, "ihs_idx"]
        ihs_points = ohlc.loc[point, "ihs_point"]
        levels = [(ohlc.index[int(i)], v) for i, v in zip(ihs_idx, ihs_points)]
        hs_points = pd.Series(index=window_data.index, dtype=float)
        for x, y in levels:
            hs_points.loc[x] = y

        addplot = [
            mpf.make_addplot(hs_points, type='scatter', markersize=200, marker='v', color='r', label='Pattern Points')
        ]

        img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_inverse-hs_{window_data.index[-1].strftime('%Y%m%d_%H%M%S')}.png")
        mpf.plot(window_data, type='candle', addplot=addplot, alines=dict(alines=levels, colors=['purple'], alpha=0.5, linewidths=20),
                title=f"Inverse Head and Shoulders for {symbol} at {window_data.index[-1]}",
                style='binance', savefig=img_path, ylabel='Price (USDT)')

        results.append({
            "Symbol": symbol,
            "Pattern Type": "Inverse Head and Shoulders",
            "Pattern Index": point,
            "Pattern Date": window_data.index[-1],
            "Current Price": round(window_data['close'].iloc[-1], 6),
            "Chart Path": img_path
        })

    return results

def process_symbol(symbol):
    """Process a single symbol for inverse head and shoulders patterns and generate charts."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return []

    # Apply inverse head and shoulders detection
    df = find_inverse_head_and_shoulders(df, lookback=LOOKBACK, pivot_interval=PIVOT_INTERVAL, 
                                        short_pivot_interval=SHORT_PIVOT_INTERVAL, 
                                        head_ratio_before=HEAD_RATIO_BEFORE, 
                                        head_ratio_after=HEAD_RATIO_AFTER, 
                                        upper_slmax=UPPER_SLMAX, progress=True)

    return generate_chart(df, symbol)

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
    """Scan provided symbols for inverse head and shoulders patterns with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        symbol_results = process_symbol(symbol)
        results.extend(symbol_results)
    return results

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Inverse Head and Shoulders Pattern Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    results = scan_symbols(symbols)

    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"ihs_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound patterns in {len(results)} instances across {len(set(r['Symbol'] for r in results))} symbols.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No inverse head and shoulders patterns found in the scanned symbols.")