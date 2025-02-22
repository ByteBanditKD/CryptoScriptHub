# MeanReversionScanner.py
# A cryptocurrency trading strategy scanner focused on mean reversion using CCXT (Bybit),
# implementing Bollinger Bands and RSI indicators to generate buy/sell/hold signals
# for USDT pairs only, excluding specific stablecoins and futures/options contracts

import numpy as np
import pandas as pd
import ccxt  # Unified cryptocurrency exchange API for fetching data
import ta  # Lightweight technical analysis library
import os
from datetime import datetime  # For timestamping output files

# ================ CONFIGURATIONS ================= #
EXCHANGE_NAME = "bybit"  # Use Bybit via CCXT
LOOKBACK_DAYS = 50  # Number of days to fetch historical data for analysis
INTERVAL = "1d"  # Daily interval for data (CCXT uses '1d' for daily)
RESULTS_FOLDER = "results"  # Folder to save output files
TYPE = "/USDT:USDT" # Change to /USDT for spot
# ================================================== #

exchange_class = getattr(ccxt, EXCHANGE_NAME)
exchange = exchange_class({"rateLimit": True, })

def fetch_historical_data(symbol, interval=INTERVAL, limit=LOOKBACK_DAYS):
    """
    Fetch historical OHLCV (Open, High, Low, Close, Volume) data for a given symbol
    from Bybit via CCXT.

    Args:
        symbol (str): Trading pair (e.g., "BTC/USDT")
        interval (str): Time interval for candles (e.g., '1d' for daily)
        limit (int): Number of candles to fetch

    Returns:
        pd.DataFrame: DataFrame with OHLCV data or None if there's an error
    """
    exchange.load_markets()

    # Format symbol for CCXT (e.g., "BTC/USDT" instead of "BTCUSDT")
    if symbol not in exchange.symbols:
        print(f"Error: {symbol} is not available on {EXCHANGE_NAME}. Skipping.")
        return None

    try:
        # Fetch OHLCV data (timestamp, open, high, low, close, volume)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            print(f"Error: No OHLCV data received for {symbol}. Check symbol and timeframe.")
            return None

        # Convert to DataFrame with lowercase column names
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp to datetime and price/volume to float
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        # Ensure enough data for indicators
        if len(df) < max(LOOKBACK_DAYS, 20):  # Need at least 20 for Bollinger Bands
            print(f"Warning: Insufficient data for {symbol}. Skipping. Data length: {len(df)}")
            return None
        # Debug: Print timestamps to verify daily interval
        # print(f"Data for {symbol}: First timestamp {df['timestamp'].iloc[0]}, "
              # f"Last timestamp {df['timestamp'].iloc[-1]}")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol} via CCXT: {e}")
        return None


def bollinger_bands(data, period=20):
    """
    Calculate Bollinger Bands for the given price data using the ta library.

    Args:
        data (pd.DataFrame): DataFrame with 'close' prices
        period (int): Lookback period for Bollinger Bands (default: 20)

    Returns:
        pd.DataFrame: DataFrame with added 'upper_band', 'middle_band', and 'lower_band' columns
    """
    # Use ta library to calculate Bollinger Bands (middle = SMA, upper/lower = ±2 standard deviations)
    bb_indicator = ta.volatility.BollingerBands(close=data['close'], window=period, window_dev=2)
    data['upper_band'] = bb_indicator.bollinger_hband()
    data['middle_band'] = bb_indicator.bollinger_mavg()
    data['lower_band'] = bb_indicator.bollinger_lband()
    # Debug: Print recent Bollinger Band values for verification
    # print(f"Bollinger Bands for {data.index[-1]}: Upper {data['upper_band'].iloc[-1]:.2f}, "
          # f"Middle {data['middle_band'].iloc[-1]:.2f}, Lower {data['lower_band'].iloc[-1]:.2f}")
    return data


def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) for the given price data using the ta library.

    Args:
        data (pd.DataFrame): DataFrame with 'close' prices
        period (int): Lookback period for RSI (default: 14)

    Returns:
        pd.DataFrame: DataFrame with added 'RSI' column
    """
    rsi_indicator = ta.momentum.RSIIndicator(close=data['close'], window=period)
    data['RSI'] = rsi_indicator.rsi()
    # Debug: Print recent RSI value for verification
    # print(f"RSI for {data.index[-1]}: {data['RSI'].iloc[-1]:.2f}")
    return data


def mean_reversion_strategy(data):
    """
    Implement a mean reversion strategy using Bollinger Bands.

    Args:
        data (pd.DataFrame): DataFrame with Bollinger Bands and 'close' price

    Returns:
        str: Trading signal ("Buy", "Sell", or "Hold") and associated lower band value
    """
    last_close = data['close'].iloc[-1]  # Most recent closing price
    middle_band = data['middle_band'].iloc[-1]  # Middle Bollinger Band
    upper_band = data['upper_band'].iloc[-1]  # Upper Bollinger Band
    lower_band = data['lower_band'].iloc[-1]  # Lower Bollinger Band

    # Buy signal if price is below lower band (potential oversold)
    if last_close < lower_band:
        return "Buy (below lower band)", lower_band
    # Sell signal if price is above upper band (potential overbought)
    elif last_close > upper_band:
        return "Sell (above upper band)", lower_band
    # Hold signal if price is within bands
    else:
        return "Hold (within bands)", lower_band


def rsi_strategy(data):
    """
    Implement an RSI-based trading strategy.

    Args:
        data (pd.DataFrame): DataFrame with 'RSI' column

    Returns:
        str: Trading signal ("Buy", "Sell", or "Hold") and associated RSI value
    """
    last_rsi = data['RSI'].iloc[-1]  # Most recent RSI value

    # Buy signal if RSI indicates oversold (below 30)
    if last_rsi < 30:
        return "Buy (RSI Oversold)", last_rsi
    # Sell signal if RSI indicates overbought (above 70)
    elif last_rsi > 70:
        return "Sell (RSI Overbought)", last_rsi
    # Hold signal if RSI is neutral
    else:
        return "Hold (RSI Neutral)", last_rsi


def process_symbol(symbol):
    """
    Process a single symbol to generate trading signals using mean reversion and RSI strategies,
    including additional validation data.

    Args:
        symbol (str): Trading pair (e.g., "BTC/USDT") in CCXT format

    Returns:
        dict: Dictionary with symbol, strategy signals, lower Bollinger Band, RSI value, and current price,
              or None if data fetch fails
    """
    # Fetch historical data
    data = fetch_historical_data(symbol, interval=INTERVAL, limit=LOOKBACK_DAYS)
    if data is None:
        return None

    # Ensure data has timestamps as index for consistency
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)

    # Calculate technical indicators for mean reversion analysis
    data = bollinger_bands(data)
    data = calculate_rsi(data)

    # Get current price (most recent closing price)
    current_price = data['close'].iloc[-1]

    # Apply trading strategies and get additional values
    mean_reversion_signal, lower_bb_value, upper_bb_value = mean_reversion_strategy(data)
    rsi_signal, rsi_value = rsi_strategy(data)

    # Optional: Validate signals for consistency (e.g., require both to indicate Buy/Sell or Hold)
    final_signal = "Hold (Inconclusive)"  # Default if signals conflict
    if mean_reversion_signal.startswith("Buy") and rsi_signal.startswith("Buy"):
        final_signal = "Buy (Confirmed)"
    elif mean_reversion_signal.startswith("Sell") and rsi_signal.startswith("Sell"):
        final_signal = "Sell (Confirmed)"
    elif mean_reversion_signal.startswith("Hold") and rsi_signal.startswith("Hold"):
        final_signal = "Hold (Confirmed)"

    return {
        "Symbol": symbol,
        "Trading Signal": final_signal,  # Combined signal for clarity
        "Mean Reversion Signal": mean_reversion_signal,
        "RSI Signal": rsi_signal,
        "Lower Bollinger Band": round(lower_bb_value, 6),  # Rounded for readability
        "Upper Bollinger Band": round(upper_bb_value, 6)
        "RSI Value": round(rsi_value, 2),  # Rounded for readability
        "Current Price": round(current_price, 6)  # Rounded for readability
    }


def fetch_symbols():
    """
    Fetch all linear contract symbols available on Bybit via CCXT, filtering for USDT pairs
    and excluding specific stablecoins and futures/options contracts.

    Returns:
        list: List of filtered symbol strings (e.g., ["BTC/USDT", "ETH/USDT"])
    """
    exchange.load_markets()

    # Filter symbols ending with "USDT" and exclude specific stablecoins and futures/options
    excluded_symbols = ["USDC/USDT", "USDE/USDT", "USTC/USDT"]
    symbols = [
        symbol for symbol in exchange.symbols
        if symbol.endswith(TYPE) and 
        not any(excluded in symbol for excluded in excluded_symbols) and 
        not ("/" in symbol and ("-" in symbol or symbol.split("/")[0][-1].isdigit()))
    ]
    return symbols


def process_all_symbols():
    """
    Process all filtered Bybit symbols, generate trading signals, and collect results
    for mean reversion analysis.

    Returns:
        list: List of dictionaries containing trading signals and validation data for each filtered symbol
    """
    symbols = fetch_symbols()
    if not symbols:
        print("No USDT symbols fetched (excluding stablecoins and futures/options). Exiting.")
        return []

    results = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        result = process_symbol(symbol)
        if result:
            results.append(result)

    return results


if __name__ == "__main__":
    # Create results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Process all filtered symbols for mean reversion signals
    results = process_all_symbols()

    # Save results to CSV in the results folder with a timestamp
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_FOLDER, f"mean_reversion_results_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Display results
        print("Mean Reversion Scanner Results:")
        print(results_df)