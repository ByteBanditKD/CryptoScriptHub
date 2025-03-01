#!/usr/bin/env python
"""
CryptoTA_Multiple.py
---------------------
This script fetches cryptocurrency price data from Bybit linear perpetual futures using CCXT and calculates various technical indicators
including Bollinger Bands, RSI, MFI, CCI, ATR, SMA, and OBV. It also analyzes RSI divergence patterns.
Users can specify multiple symbols to scan (all Bybit futures, from a CSV file, or a predefined list) with a fixed limit of 100 candles on daily timeframe.
Results are saved to a CSV file in the 'results' directory.

Requirements:
- pip install ccxt pandas numpy matplotlib ta
- Python 3.7+

Usage:
- Run the script: python CryptoTA_Standalone.py
- Follow prompts to choose symbol input method (all futures, CSV, or predefined list)
- Results are saved in 'results/ta_scan_[timestamp].csv'

Source: Inspired by https://alpaca.markets/learn/algorithmic-trading-chart-pattern-python/
"""

import ccxt
import pandas as pd
import os
from datetime import datetime
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import MFIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import CCIIndicator
from ta.volume import OnBalanceVolumeIndicator
from tqdm import tqdm

# ================ CONFIGURATIONS ================= #
EXCHANGE_NAME = "bybit"
INTERVAL = "1d"   # Daily interval
LIMIT = 100       # Fixed number of candles for technical analysis
RESULTS_FOLDER = "results"
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

def calculate_technical_indicators(df):
    """Calculate technical indicators for the given DataFrame."""
    df = dropna(df)
    
    # Bollinger Bands
    indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI
    indicator_rsi = RSIIndicator(close=df['close'])
    df['rsi'] = indicator_rsi.rsi()
    
    # MFI
    indicator_mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
    df['mfi'] = indicator_mfi.money_flow_index()
    
    # CCI
    indicator_cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['cci'] = indicator_cci.cci()
    
    # ATR
    indicator_atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['atr'] = indicator_atr.average_true_range()
    
    # SMA
    indicator_sma = SMAIndicator(close=df['close'], window=50)
    df['sma50'] = indicator_sma.sma_indicator()
    
    # OBV
    indicator_obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['obv'] = indicator_obv.on_balance_volume()
    
    # RSI Divergence
    df['rsi_divergence'] = df['rsi'].diff()
    df['price_divergence'] = df['close'].diff()
    df['rsi_divergence_type'] = df.apply(
        lambda row: 'Bullish' if row['rsi_divergence'] > 0 and row['price_divergence'] < 0 
        else ('Bearish' if row['rsi_divergence'] < 0 and row['price_divergence'] > 0 
        else 'None'), 
        axis=1
    )
    
    return df

def process_symbol(symbol):
    """Process a single symbol for technical analysis and save results."""
    df = fetch_historical_data(symbol)
    if df is None or df.empty:
        return None

    # Calculate technical indicators
    df_ta = calculate_technical_indicators(df)
    
    # Get the most recent data point for technical analysis
    latest = df_ta.iloc[-1]
    
    # Calculate OBV trend over last 5 bars
    obv_trend = "Rising" if df_ta['obv'].iloc[-5:].diff().mean() > 0 else "Falling"
    
    # Determine Bollinger Bands position
    bb_range = latest['bb_bbh'] - latest['bb_bbl']
    bb_proximity_threshold = bb_range * 0.1  # Consider "near" if within 10% of band width
    
    if latest['close'] > latest['bb_bbh']:
        bb_position = "Above Upper"
    elif latest['close'] < latest['bb_bbl']:
        bb_position = "Below Lower"
    elif latest['close'] >= (latest['bb_bbh'] - bb_proximity_threshold):
        bb_position = "Near Upper Band"
    elif latest['close'] <= (latest['bb_bbl'] + bb_proximity_threshold):
        bb_position = "Near Lower Band"
    else:
        bb_position = f"Within Bands ({'Above' if latest['close'] > latest['bb_bbm'] else 'Below'} Middle)"
    
    # Prepare technical analysis results
    ta_results = {
        "Symbol": symbol,
        "Timeframe": INTERVAL,
        "Timestamp": latest.name,
        "Current Price": round(latest['close'], 6),
        "High": round(latest['high'], 6),
        "Low": round(latest['low'], 6),
        "Volume": round(latest['volume'], 2),
        "RSI": round(latest['rsi'], 2),
        "MFI": round(latest['mfi'], 2),
        "CCI": round(latest['cci'], 2),
        "ATR": round(latest['atr'], 6),
        "SMA50": round(latest['sma50'], 6),
        "OBV Trend": obv_trend,
        "BB Upper": round(latest['bb_bbh'], 6),
        "BB Middle": round(latest['bb_bbm'], 6),
        "BB Lower": round(latest['bb_bbl'], 6),
        "BB Position": bb_position,
        "RSI Divergence Type": latest['rsi_divergence_type']
    }
    
    # Print technical analysis summary
    print(f"\nTechnical Analysis Summary for {symbol} ({INTERVAL} Timeframe)")
    print(f"As of {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}:")
    print("-" * 60)
    print(f"Current Price: ${ta_results['Current Price']}")
    print(f"{INTERVAL} Range: ${ta_results['Low']} - ${ta_results['High']}")
    print(f"Volume: {ta_results['Volume']}")
    print("\nKey Indicators:")
    print(f"RSI (14): {ta_results['RSI']} {'(Overbought)' if ta_results['RSI'] > 70 else '(Oversold)' if ta_results['RSI'] < 30 else ''}")
    print(f"MFI (14): {ta_results['MFI']} {'(Overbought)' if ta_results['MFI'] > 80 else '(Oversold)' if ta_results['MFI'] < 20 else ''}")
    print(f"CCI: {ta_results['CCI']}")
    print(f"ATR (14): ${ta_results['ATR']}")
    print(f"SMA (50): ${ta_results['SMA50']} ({'Above' if ta_results['Current Price'] > ta_results['SMA50'] else 'Below'} SMA)")
    print(f"OBV (5-bar trend): {ta_results['OBV Trend']}")
    print("\nBollinger Bands (20,2):")
    print(f"Upper: ${ta_results['BB Upper']}")
    print(f"Middle: ${ta_results['BB Middle']}")
    print(f"Lower: ${ta_results['BB Lower']}")
    print(f"Position: {ta_results['BB Position']}")
    print("\nDivergence Analysis:")
    print(f"RSI Divergence: {ta_results['RSI Divergence Type']}")
    print("-" * 60)
    
    return ta_results

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
    """Scan provided symbols for technical analysis, saving results to a CSV file with a progress bar."""
    ta_results = []
    
    for symbol in tqdm(symbols, desc="Scanning symbols"):
        print(f"Processing {symbol}...")
        ta_result = process_symbol(symbol)
        if ta_result:
            ta_results.append(ta_result)
    
    if ta_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"ta_scan_{timestamp}.csv")
        ta_df = pd.DataFrame(ta_results)
        ta_df.to_csv(output_file, index=False)
        print(f"\nFound technical analysis for {len(ta_results)} symbols.")
        print(f"Technical analysis results saved to {output_file}")
        print("\nTechnical Analysis Results:")
        print(ta_df)
    else:
        print("\nNo technical analysis results found in the scanned symbols.")

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print("Welcome to Crypto Technical Analyzer (Bybit Futures)!")
    symbols, source = get_symbols_choice()
    print(f"Scanning {len(symbols)} symbols from {source}...")

    scan_symbols(symbols)