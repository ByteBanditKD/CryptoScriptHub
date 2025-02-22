# Import required libraries
import ccxt              # For cryptocurrency exchange data
import pandas as pd      # For data manipulation and analysis
import numpy as np      # For numerical operations

class Sup_Res_Finder:
    """Class to identify support and resistance levels in trading data"""
    
    def isSupport(self, df, i):
        """
        Check if the current point is a support level
        Args:
            df (DataFrame): Price data
            i (int): Index to check
        Returns:
            bool: True if support level detected
        """
        support = (df['Low'][i] < df['Low'][i-1] and 
                  df['Low'][i] < df['Low'][i+1] and
                  df['Low'][i+1] < df['Low'][i+2] and 
                  df['Low'][i-1] < df['Low'][i-2])
        return support

    def isResistance(self, df, i):
        """
        Check if the current point is a resistance level
        Args:
            df (DataFrame): Price data
            i (int): Index to check
        Returns:
            bool: True if resistance level detected
        """
        resistance = (df['High'][i] > df['High'][i-1] and 
                     df['High'][i] > df['High'][i+1] and
                     df['High'][i+1] > df['High'][i+2] and 
                     df['High'][i-1] > df['High'][i-2])
        return resistance
        
    def find_levels(self, df):
        """
        Find all support and resistance levels in the data
        Args:
            df (DataFrame): Price data
        Returns:
            list: List of tuples (index, price) of levels
        """
        levels = []
        # Calculate average candle size as threshold
        s = np.mean(df['High'] - df['Low'])
        
        # Check each candle (excluding first/last 2 for pattern recognition)
        for i in range(2, df.shape[0]-2):
            if self.isSupport(df, i):
                l = df['Low'][i]
                # Only add if no similar level exists within threshold
                if np.sum([abs(l-x) < s for x in levels]) == 0:
                    levels.append((i, l))
            
            elif self.isResistance(df, i):
                l = df['High'][i]
                # Only add if no similar level exists within threshold
                if np.sum([abs(l-x) < s for x in levels]) == 0:
                    levels.append((i, l))
                    
        return levels

# Main execution
def main():
    # Get user input for trading parameters
    symbol = input("Enter the trading symbol (e.g., BTC/USDT): ")
    timeframe = input("Enter the timeframe (e.g., 1h, 4h, 1d): ")
    limit = int(input("Enter the number of candles to fetch: "))

    # Initialize Binance exchange connection
    exchange = ccxt.binance()

    try:
        # Fetch OHLCV (Open, High, Low, Close, Volume) data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert raw data to DataFrame with proper column names
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')  # Convert milliseconds to datetime
        
        # Find support and resistance levels
        sr_finder = Sup_Res_Finder()
        levels = sr_finder.find_levels(df)
        
        # Separate support and resistance levels
        supports = []
        resistances = []
        
        for level in levels:
            index, price = level
            if df['Low'][index] == price:
                supports.append(price)
            else:
                resistances.append(price)
        
        # Sort levels in ascending order
        supports.sort()
        resistances.sort()
        
        # Display results
        print("\nSupport/Resistance Levels (Sorted):")
        print("\nSupport Levels:")
        for price in supports:
            index = df[df['Low'] == price].index[0]
            print(f"Support at {price:.4f} on {df['Timestamp'][index]}")
        
        print("\nResistance Levels:")
        for price in resistances:
            index = df[df['High'] == price].index[0]
            print(f"Resistance at {price:.4f} on {df['Timestamp'][index]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()