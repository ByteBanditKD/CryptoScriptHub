# CryptoTA_API.py - Modified version for API usage
"""
Crypto Technical Analyzer - API Version
This script fetches cryptocurrency price data from Binance using CCXT and calculates various technical indicators
including Bollinger Bands, RSI, MFI, CCI, ATR, SMA, and OBV. It also analyzes RSI divergence patterns.
This version reads parameters from environment variables for API integration.
"""

import ccxt
import pandas as pd
import os
import json
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
    # Initialize exchange through CCXT (try multiple exchanges for better availability)
    exchanges_to_try = ['bybit', 'okx', 'kucoin', 'binance']
    exchange = None
    
    for exchange_name in exchanges_to_try:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({'enableRateLimit': True})
            # Test the connection
            exchange.load_markets()
            print(f"Using {exchange_name} exchange")
            break
        except Exception as e:
            print(f"Failed to connect to {exchange_name}: {e}")
            continue
    
    if not exchange:
        raise Exception("Could not connect to any cryptocurrency exchange")
    
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
    
    indicator_mfi = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    df['mfi'] = indicator_mfi.money_flow_index()
    
    indicator_cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['cci'] = indicator_cci.cci()
    
    indicator_atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['atr'] = indicator_atr.average_true_range()
    
    indicator_sma50 = SMAIndicator(close=df['close'], window=50)
    df['sma50'] = indicator_sma50.sma_indicator()
    
    indicator_obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['obv'] = indicator_obv.on_balance_volume()
    
    # RSI Divergence Analysis
    df['rsi_divergence_type'] = 'None'
    
    # Simple divergence detection (last 10 bars)
    if len(df) >= 10:
        recent_data = df.tail(10)
        price_trend = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        rsi_trend = recent_data['rsi'].iloc[-1] - recent_data['rsi'].iloc[0]
        
        if price_trend > 0 and rsi_trend < 0:
            df.loc[df.index[-1], 'rsi_divergence_type'] = 'Bearish'
        elif price_trend < 0 and rsi_trend > 0:
            df.loc[df.index[-1], 'rsi_divergence_type'] = 'Bullish'
    
    # Get latest values
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
    
    # Create analysis result
    analysis_result = {
        'symbol': symbol,
        'timeframe': timeframe_label,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
        'current_price': float(latest['close']),
        'daily_range': {
            'low': float(latest['low']),
            'high': float(latest['high'])
        },
        'volume': float(latest['volume']),
        'indicators': {
            'rsi': float(latest['rsi']),
            'rsi_signal': 'Overbought' if latest['rsi'] > 70 else 'Oversold' if latest['rsi'] < 30 else 'Neutral',
            'mfi': float(latest['mfi']),
            'mfi_signal': 'Overbought' if latest['mfi'] > 80 else 'Oversold' if latest['mfi'] < 20 else 'Neutral',
            'cci': float(latest['cci']),
            'atr': float(latest['atr']),
            'sma50': float(latest['sma50']),
            'sma_position': 'Above' if latest['close'] > latest['sma50'] else 'Below',
            'obv_trend': obv_trend
        },
        'bollinger_bands': {
            'upper': float(latest['bb_bbh']),
            'middle': float(latest['bb_bbm']),
            'lower': float(latest['bb_bbl']),
            'position': bb_position
        },
        'divergence': {
            'rsi_divergence': latest['rsi_divergence_type']
        }
    }
    
    # Print formatted output for console
    print(f"\nTechnical Analysis Summary for {symbol} ({timeframe_label} Timeframe)")
    print(f"As of {analysis_result['timestamp']}:")
    print("-" * 60)
    print(f"Current Price: ${analysis_result['current_price']:,.4f}")
    print(f"{timeframe_label} Range: ${analysis_result['daily_range']['low']:,.4f} - ${analysis_result['daily_range']['high']:,.4f}")
    print(f"Volume: {analysis_result['volume']:,.2f}")
    print("\nKey Indicators:")
    print(f"RSI (14): {analysis_result['indicators']['rsi']:.2f} ({analysis_result['indicators']['rsi_signal']})")
    print(f"MFI (14): {analysis_result['indicators']['mfi']:.2f} ({analysis_result['indicators']['mfi_signal']})")
    print(f"CCI: {analysis_result['indicators']['cci']:.2f}")
    print(f"ATR (14): ${analysis_result['indicators']['atr']:,.4f}")
    print(f"SMA (50): ${analysis_result['indicators']['sma50']:,.4f} ({analysis_result['indicators']['sma_position']} SMA)")
    print(f"OBV (5-bar trend): {analysis_result['indicators']['obv_trend']}")
    print("\nBollinger Bands (20,2):")
    print(f"Upper: ${analysis_result['bollinger_bands']['upper']:,.4f}")
    print(f"Middle: ${analysis_result['bollinger_bands']['middle']:,.4f}")
    print(f"Lower: ${analysis_result['bollinger_bands']['lower']:,.4f}")
    print(f"Position: {analysis_result['bollinger_bands']['position']}")
    print("\nDivergence Analysis:")
    print(f"RSI Divergence: {analysis_result['divergence']['rsi_divergence']}")
    print("-" * 60)
    
    # Save results to JSON file for API consumption
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"crypto_ta_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return analysis_result

if __name__ == "__main__":
    # Check for environment variables (for API usage)
    symbol = os.environ.get('CRYPTO_PARAM_SYMBOL', '').strip()
    timeframe = os.environ.get('CRYPTO_PARAM_TIMEFRAME', '').strip().lower()
    
    # If no environment variables, get user inputs
    if not symbol:
        symbol = input("Please enter the cryptocurrency symbol (e.g., BTC): ")
    if not timeframe:
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