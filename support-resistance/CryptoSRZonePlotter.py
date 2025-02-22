# Import required libraries
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin

# Initialize Binance exchange (no API key needed for public data)
exchange = ccxt.binance()

# Prompt user for trading pair symbol
symbol = input("Enter the trading pair symbol (e.g., BTC/USDT): ")

# Define timeframe and limit
timeframe = '1d'  # Daily interval
limit = 180      # Number of data points to retrieve

# Fetch OHLCV data from Binance using CCXT
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# Convert to Pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Convert columns to numeric data types
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Remove rows with missing values
df.dropna(inplace=True)

# Convert timestamps to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

# Set Timestamp as index
df = df.set_index('Timestamp')

# Calculate support and resistance levels
df_low = df['Low']
df_high = df['High']
support = argrelmin(df_low.values, order=5)
resistance = argrelmax(df_high.values, order=5)
support_prices = df_low.iloc[support[0]]
resistance_prices = df_high.iloc[resistance[0]]

# Get current price
current_price = df['Close'].iloc[-1]

# Filter levels relative to current price
support_prices = support_prices[support_prices < current_price]
resistance_prices = resistance_prices[resistance_prices > current_price]

# Print support and resistance prices
print("Support prices:")
print(support_prices)
print("\nResistance prices:")
print(resistance_prices)

# Plot 1: Individual Levels
plt.figure(figsize=(15, 9))
plt.plot(df.index, df['Close'], color='blue', label='Closing Price')

# Plot support prices
for date, price in support_prices.items():
    plt.hlines(price, xmin=df.index.min(), xmax=df.index.max(), 
               colors='green', linestyles='--', label='Support')
    plt.text(df.index.max(), price, f'{price:.2f}', 
             ha='left', va='center', color='green')

# Plot resistance prices
for date, price in resistance_prices.items():
    plt.hlines(price, xmin=df.index.min(), xmax=df.index.max(), 
               colors='red', linestyles='--', label='Resistance')
    plt.text(df.index.max(), price, f'{price:.2f}', 
             ha='left', va='center', color='red')

plt.legend()
plt.title(f'Market Closing Prices with Support & Resistance Levels for {symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('Closing_Prices_with_SR_Levels.png')
plt.close()

# Group support prices into zones
support_zones = []
if len(support_prices) > 0:
    support_zone = [list(support_prices.items())[0]]
    for date, price in list(support_prices.items())[1:]:
        if (date - support_zone[-1][0] <= pd.Timedelta(days=7) and 
            abs(price - support_zone[-1][1]) <= 750):
            support_zone.append((date, price))
        else:
            support_zones.append(support_zone)
            support_zone = [(date, price)]
    support_zones.append(support_zone)

# Group resistance prices into zones
resistance_zones = []
if len(resistance_prices) > 0:
    resistance_zone = [list(resistance_prices.items())[0]]
    for date, price in list(resistance_prices.items())[1:]:
        if (date - resistance_zone[-1][0] <= pd.Timedelta(days=7) and 
            abs(price - resistance_zone[-1][1]) <= 750):
            resistance_zone.append((date, price))
        else:
            resistance_zones.append(resistance_zone)
            resistance_zone = [(date, price)]
    resistance_zones.append(resistance_zone)

# Plot 2: Zones
plt.figure(figsize=(15, 9))
plt.plot(df.index, df['Close'], color='blue', label='Closing Price')

# Plot support zones
for zone in support_zones:
    start_date, start_price = zone[0]
    end_date, _ = zone[-1]
    plt.hlines(start_price, xmin=start_date, xmax=end_date, 
               colors='green', linestyles='--', label='Support Zone')
    plt.text(df.index.max(), start_price, f'{start_price:.2f}', 
             ha='left', va='center', color='green')

# Plot resistance zones
for zone in resistance_zones:
    start_date, start_price = zone[0]
    end_date, _ = zone[-1]
    plt.hlines(start_price, xmin=start_date, xmax=end_date, 
               colors='red', linestyles='--', label='Resistance Zone')
    plt.text(df.index.max(), start_price, f'{start_price:.2f}', 
             ha='left', va='center', color='red')

plt.legend()
plt.title(f'Market Closing Prices with Support & Resistance Zones for {symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('Closing_Prices_with_SR_Zones.png')
plt.close()