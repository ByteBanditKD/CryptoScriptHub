# CryptoTA.py
"""
Crypto Technical Analyzer
This script fetches cryptocurrency price data from Binance using CCXT and calculates various technical indicators
including Bollinger Bands, RSI, MFI, CCI, ATR, SMA, and OBV. It also analyzes RSI divergence patterns.
Users can specify the timeframe (e.g., 1h, 4h, 1d) for analysis, with a fixed limit of 100 candles.
"""

import ccxt
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import MFIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import CCIIndicator
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime

def get_crypto_data(symbol, timeframe):
    # Initialize Binance exchange through CCXT
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Convert symbol to Binance format
    symbol = symbol.upper() + '/USDT'
    
    # Set fixed limit for all timeframes (enough for 50-period SMA + buffer)
    limit = 100
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # Create DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = dropna(df)
    
    # Calculate technical indicators
    indicator_bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    indicator_rsi = RSIIndicator(close=df['close'])
    df['rsi'] = indicator_rsi.rsi()
    
    indicator_mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
    df['mfi'] = indicator_mfi.money_flow_index()
    
    indicator_cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['cci'] = indicator_cci.cci()
    
    indicator_atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['atr'] = indicator_atr.average_true_range()
    
    indicator_sma = SMAIndicator(close=df['close'], window=50)
    df['sma50'] = indicator_sma.sma_indicator()
    
    # Add OBV indicator
    indicator_obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['obv'] = indicator_obv.on_balance_volume()
    
    df['rsi_divergence'] = df['rsi'].diff()
    df['price_divergence'] = df['close'].diff()
    df['rsi_divergence_type'] = df.apply(
        lambda row: 'Bullish' if row['rsi_divergence'] > 0 and row['price_divergence'] < 0 
        else ('Bearish' if row['rsi_divergence'] < 0 and row['price_divergence'] > 0 
        else 'None'), 
        axis=1
    )
    
    # Get the most recent data point
    latest = df.iloc[-1]
    
    # Calculate OBV trend over last 5 bars
    obv_trend = "Rising" if df['obv'].iloc[-5:].diff().mean() > 0 else "Falling"
    
    # Adjust range label based on timeframe
    timeframe_label = {
        '1m': '1-Minute', '5m': '5-Minute', '15m': '15-Minute', '30m': '30-Minute',
        '1h': '1-Hour', '2h': '2-Hour', '4h': '4-Hour', '6h': '6-Hour', 
        '12h': '12-Hour', '1d': 'Daily'
    }.get(timeframe, '1-Hour')  # Default to 1-Hour if invalid
    
    # Determine Bollinger Bands position with more detail
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
    
    # Format and print technical analysis summary with 4 decimal places
    print(f"\nTechnical Analysis Summary for {symbol} ({timeframe_label} Timeframe)")
    print(f"As of {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}:")
    print("-" * 60)
    print(f"Current Price: ${latest['close']:,.4f}")
    print(f"{timeframe_label} Range: ${latest['low']:,.4f} - ${latest['high']:,.4f}")
    print(f"Volume: {latest['volume']:,.2f}")
    print("\nKey Indicators:")
    print(f"RSI (14): {latest['rsi']:.2f} {'(Overbought)' if latest['rsi'] > 70 else '(Oversold)' if latest['rsi'] < 30 else ''}")
    print(f"MFI (14): {latest['mfi']:.2f} {'(Overbought)' if latest['mfi'] > 80 else '(Oversold)' if latest['mfi'] < 20 else ''}")
    print(f"CCI: {latest['cci']:.2f}")
    print(f"ATR (14): ${latest['atr']:,.4f}")
    print(f"SMA (50): ${latest['sma50']:,.4f} ({'Above' if latest['close'] > latest['sma50'] else 'Below'} SMA)")
    print(f"OBV (5-bar trend): {obv_trend}")
    print("\nBollinger Bands (20,2):")
    print(f"Upper: ${latest['bb_bbh']:,.4f}")
    print(f"Middle: ${latest['bb_bbm']:,.4f}")
    print(f"Lower: ${latest['bb_bbl']:,.4f}")
    print(f"Position: {bb_position}")
    print("\nDivergence Analysis:")
    print(f"RSI Divergence: {latest['rsi_divergence_type']}")
    print("-" * 60)

if __name__ == "__main__":
    # Get user inputs
    symbol = input("Please enter the cryptocurrency symbol (e.g., BTC): ")
    print("Available timeframes: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d")
    timeframe = input("Please enter the timeframe (e.g., 1h): ").lower()
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']
    if timeframe not in valid_timeframes:
        print(f"Invalid timeframe. Using default '1h' instead.")
        timeframe = '1h'
    
    try:
        get_crypto_data(symbol, timeframe)
    except ccxt.ExchangeError as e:
        print(f"Exchange error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")