"""
OversoldScanner.py
------------------
Disclaimer: These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions, where applicable, are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

This script scans Bybit spot market USDT pairs for coins that are oversold (RSI < 30) and have closed below their lower Bollinger Band.
It fetches OHLCV data using CCXT, calculates RSI and Bollinger Bands with the ta library, generates candlestick charts with Bollinger Bands,
and saves results to a CSV file. Users can choose to scan all USDT pairs, use a CSV file, or a predefined symbol list.

Requirements:
- pip install ccxt pandas numpy ta matplotlib mplfinance tqdm
- Python 3.7+
- Optional: pairs.csv file with a 'Symbol' column (e.g., 'BTCUSDT')

Usage:
- Run the script: python OversoldScanner.py
- Follow prompts to choose input method (all pairs, CSV, or symbol list)
- Results are saved in 'results/oversold_scan_[timestamp].csv'
- Charts are saved in 'img/[symbol]_oversold.png'
"""

import numpy as np
import pandas as pd
import ccxt
import ta
import os
from datetime import datetime
import mplfinance as mpf
from tqdm import tqdm

# ================ CONFIGURATIONS ================= #
EXCHANGE_NAME = "bybit"
LOOKBACK_DAYS = 50
INTERVAL = "1d"
RESULTS_FOLDER = "results"
IMG_DIR = "img"
RSI_THRESHOLD = 30
CSV_FILE = "pairs.csv"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"]
# ================================================== #

exchange_class = getattr(ccxt, EXCHANGE_NAME)
exchange = exchange_class({"enableRateLimit": True})

def adapt_symbol(symbol):
    """Convert symbol format to Bybit spot-compatible CCXT format (e.g., 'BTCUSDT' to 'BTC/USDT')."""
    symbol = symbol.strip().replace('\xa0', '')
    if '/' not in symbol:  # If no '/', assume format like BTCUSDT
        base = symbol[:-4].upper()
        quote = symbol[-4:].upper()
        return f"{base}/{quote}"
    return symbol.upper()  # Already in BTC/USDT format

def fetch_historical_data(symbol, interval=INTERVAL, limit=LOOKBACK_DAYS):
    """Fetch historical OHLCV data for a given symbol from Bybit spot market."""
    exchange.load_markets()
    adapted_symbol = adapt_symbol(symbol)
    if adapted_symbol not in exchange.symbols:
        print(f"Error: {adapted_symbol} not available on {EXCHANGE_NAME} spot market. Skipping.")
        return None

    try:
        ohlcv = exchange.fetch_ohlcv(adapted_symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            print(f"Error: No data received for {adapted_symbol}.")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        if len(df) < max(LOOKBACK_DAYS, 20):
            print(f"Warning: Insufficient data for {adapted_symbol}. Skipping. Data length: {len(df)}")
            return None
        
        return df
    except Exception as e:
        print(f"Error fetching data for {adapted_symbol}: {e}")
        return None

def calculate_indicators(data):
    """Calculate RSI and Bollinger Bands for the given price data."""
    rsi_indicator = ta.momentum.RSIIndicator(close=data['close'], window=14)
    data['RSI'] = rsi_indicator.rsi()

    bb_indicator = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['upper_band'] = bb_indicator.bollinger_hband()
    data['middle_band'] = bb_indicator.bollinger_mavg()
    data['lower_band'] = bb_indicator.bollinger_lband()

    return data

def is_oversold_below_lower_band(data):
    """Check if the latest close is below the lower Bollinger Band and RSI < 30."""
    last_close = data['close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    last_lower_band = data['lower_band'].iloc[-1]

    return last_rsi < RSI_THRESHOLD and last_close < last_lower_band

def generate_chart(data, symbol):
    """Generate and save a candlestick chart with Bollinger Bands."""
    addplot = [
        mpf.make_addplot(data['upper_band'], color='blue', linestyle='--', label='Upper BB'),
        mpf.make_addplot(data['middle_band'], color='gray', linestyle='--', label='Middle BB'),
        mpf.make_addplot(data['lower_band'], color='red', linestyle='--', label='Lower BB')
    ]
    img_path = os.path.join(IMG_DIR, f"{symbol.replace('/', '_')}_oversold.png")
    mpf.plot(data, type='candle', addplot=addplot, title=f"{symbol} - Oversold (RSI < 30, Below Lower BB)",
             style='binance', savefig=img_path, ylabel='Price (USDT)')
    return img_path

def process_symbol(symbol):
    """Process a single symbol for oversold conditions and generate a chart if applicable."""
    data = fetch_historical_data(symbol)
    if data is None:
        return None

    data = calculate_indicators(data)
    
    if is_oversold_below_lower_band(data):
        chart_path = generate_chart(data, symbol)
        return {
            "Symbol": symbol,
            "Current Price": round(data['close'].iloc[-1], 6),
            "RSI": round(data['RSI'].iloc[-1], 2),
            "Lower Bollinger Band": round(data['lower_band'].iloc[-1], 6),
            "Middle Bollinger Band": round(data['middle_band'].iloc[-1], 6),
            "Upper Bollinger Band": round(data['upper_band'].iloc[-1], 6),
            "Chart Path": chart_path
        }
    return None

def load_symbols_from_csv(csv_file):
    """Load trading symbols from a CSV file."""
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Using default symbol list instead.")
        return None
    return pd.read_csv(csv_file)['Symbol'].tolist()

def fetch_symbols():
    """Fetch all spot USDT pairs from Bybit, excluding stablecoins and futures."""
    exchange.load_markets()
    excluded_symbols = ["USDC/USDT", "USDE/USDT", "USTC/USDT"]
    symbols = [
        symbol for symbol in exchange.symbols
        if symbol.endswith("/USDT") and 
        not any(excluded in symbol for excluded in excluded_symbols) and
        ":" not in symbol
    ]
    return symbols

def get_symbols_choice():
    """Prompt user to choose symbol input method."""
    print("\nChoose how to select symbols:")
    print("1. Scan all Bybit USDT spot pairs")
    print("2. Use pairs.csv file")
    print("3. Use predefined symbol list (BTCUSDT, ETHUSDT, XRPUSDT, ADAUSDT)")
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    if choice == '1':
        return fetch_symbols(), "all Bybit USDT spot pairs"
    elif choice == '2':
        symbols = load_symbols_from_csv(CSV_FILE)
        return symbols if symbols else DEFAULT_SYMBOLS, f"CSV file ({CSV_FILE})" if symbols else "default list (CSV not found)"
    else:
        return DEFAULT_SYMBOLS, "predefined symbol list"

def scan_symbols(symbols):
    """Scan provided symbols for oversold conditions with a progress bar."""
    results = []
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        result = process_symbol(symbol)
        if result:
            results.append(result)
    return results

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Welcome to Oversold Scanner!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    # Scan symbols
    results = scan_symbols(symbols)

    # Save and display results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"oversold_scan_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nFound {len(results)} oversold coins below lower Bollinger Band.")
        print(f"Results saved to {output_file}")
        print("\nScan Results:")
        print(results_df)
    else:
        print("No coins found meeting the oversold criteria (RSI < 30 and below lower Bollinger Band).")