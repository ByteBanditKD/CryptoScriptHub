"""
Directional_Change_Analysis.py
----------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script analyzes Bybit futures cryptocurrency price data to identify tops and bottoms using the Directional Change algorithm.
It fetches OHLCV data for perpetual futures pairs, detects price extremes, plots candlestick charts with markers, and saves results to a CSV file.
A progress bar is included to show processing status when handling multiple symbols.

Requirements:
- pip install pandas numpy matplotlib mplfinance ccxt tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column containing symbols (e.g., 'BTCUSDT' for BTC/USDT:USDT futures)

Usage:
- Edit the parameters section below to set symbol, timeframe, sigma, etc.
- Run directly: python Directional_Change_Analysis.py
- Results are saved to 'results/bottom_finder.csv'
- Charts are saved in the 'img' directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import ccxt
import os
from tqdm import tqdm  # Added for progress bar

def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    """Identify tops and bottoms using the Directional Change algorithm."""
    up_zig = True  # True: seeking top, False: seeking bottom
    tmp_max = high[0]  # Temporary maximum price
    tmp_min = low[0]   # Temporary minimum price
    tmp_max_i = 0      # Index of temporary maximum
    tmp_min_i = 0      # Index of temporary minimum

    tops = []     # Stores [confirmation_index, extreme_index, price]
    bottoms = []  # Stores [confirmation_index, extreme_index, price]

    for i in range(len(close)):
        if up_zig:
            if high[i] > tmp_max:
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma:
                tops.append([i, tmp_max_i, tmp_max])
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else:
            if low[i] < tmp_min:
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma:
                bottoms.append([i, tmp_min_i, tmp_min])
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms

def get_extremes(ohlc: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Convert OHLC data into a DataFrame of extremes."""
    tops, bottoms = directional_change(ohlc['close'].to_numpy(), 
                                     ohlc['high'].to_numpy(), 
                                     ohlc['low'].to_numpy(), 
                                     sigma)
    # Fixed typo: 'bottoms' instead of 'customs'
    tops_df = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms_df = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops_df['type'] = 1    # 1 for tops
    bottoms_df['type'] = -1  # -1 for bottoms
    extremes = pd.concat([tops_df, bottoms_df]).set_index('conf_i').sort_index()
    return extremes

def fetch_data(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV data for Bybit futures using CCXT."""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Set to fetch futures data
        }
    })
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Failed to fetch data for {symbol}: {e}")

def load_symbols_from_csv(csv_file: str) -> list:
    """Load trading symbols from a CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} does not exist")
    return pd.read_csv(csv_file)['Symbol'].tolist()

def adapt_symbol(symbol: str) -> str:
    """Convert symbol format to Bybit futures-compatible CCXT format (e.g., 'BTCUSDT' to 'BTC/USDT:USDT')."""
    symbol = symbol.strip().replace('\xa0', '')  # Remove trailing spaces and non-breaking spaces
    base = symbol[:-4].upper()
    quote = symbol[-4:].upper()
    return f"{base}/{quote}:{quote}"  # Format for Bybit perpetual futures (e.g., BTC/USDT:USDT)

if __name__ == "__main__":
    # Parameters
    CSV_FILE = 'pairs.csv'        # Path to CSV file with symbols (optional)
    SYMBOL = 'BTCUSDT'           # Default symbol if no CSV (will become BTC/USDT:USDT)
    TIMEFRAME = '4h'             # Candlestick timeframe
    SIGMA = 0.02                 # Threshold for directional change (2%)
    IMG_DIR = 'img'              # Directory for charts
    RESULTS_DIR = 'results'
    RESULTS_CSV = f'{RESULTS_DIR}/bottom_finder.csv'  # Output CSV file

    # Ensure directories exist
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load symbols
    if CSV_FILE and os.path.exists(CSV_FILE):
        symbols = load_symbols_from_csv(CSV_FILE)
    else:
        symbols = [SYMBOL]
    print(f"Processing {len(symbols)} futures symbols: {symbols}")

    results = []  # Store analysis results

    # Process symbols with a progress bar
    for symbol in tqdm(symbols, desc="Analyzing symbols"):
        adapted_symbol = adapt_symbol(symbol)
        try:
            # Fetch and process data
            data = fetch_data(adapted_symbol, TIMEFRAME)
            tops, bottoms = directional_change(data['close'].to_numpy(), 
                                             data['high'].to_numpy(), 
                                             data['low'].to_numpy(), 
                                             SIGMA)

            # Get last extremes
            last_top = tops[-1] if tops else [np.nan, np.nan, np.nan]
            last_bottom = bottoms[-1] if bottoms else [np.nan, np.nan, np.nan]

            # Prepare markers for plotting
            top_markers = [np.nan] * len(data)
            bottom_markers = [np.nan] * len(data)
            if tops:
                top_markers[last_top[1]] = last_top[2]
            if bottoms:
                bottom_markers[last_bottom[1]] = last_bottom[2]

            # Create additional plots
            addplot = [
                mpf.make_addplot(top_markers, type='scatter', markersize=100, marker='^', color='g'),
                mpf.make_addplot(bottom_markers, type='scatter', markersize=100, marker='v', color='r')  # Fixed: should be bottom_markers
            ]

            # Generate and save chart
            img_path = os.path.join(IMG_DIR, f"{symbol}.png")
            mpf.plot(data, type='candle', addplot=addplot, 
                    title=f"Directional Change Analysis for {adapted_symbol} (Futures)", 
                    style='binance', savefig=img_path)

            # Collect results
            current_price = data['close'].iloc[-1]
            results.append({
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol,
                'last_top': last_top[2],
                'current_price': current_price,
                'last_bottom': last_bottom[2]
            })

        except ccxt.BadSymbol as e:
            print(f"Invalid futures symbol {symbol}: {e}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Save results to CSV
    if results:
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
        print(f"Results saved to {RESULTS_CSV}")
    else:
        print("No results to save.")