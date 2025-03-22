import yfinance as yf
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

# 1. Fetch data with extended range (5 years)
ticker = "AAPL"
end_date = pd.Timestamp("2025-03-22")
start_date = end_date - pd.DateOffset(days=5*365)  # 5 years of data

df = yf.download(
    tickers=ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True
)

if df.empty:
    raise ValueError("No data returned from yfinance.")

df = df.reset_index()
print("Columns in DataFrame:", df.columns.tolist())
print("First few rows:\n", df.head())

# 2. Flatten MultiIndex and prepare data
df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
required_columns = {'Date': 'timestamp', 'Close': 'target'}
if not set(required_columns.keys()).issubset(df.columns):
    missing = set(required_columns.keys()) - set(df.columns)
    raise ValueError(f"Missing columns in yfinance data: {missing}")

df = df.rename(columns=required_columns)
df['item_id'] = ticker
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

# Fill missing dates (e.g., weekends) with forward-fill
df = df.set_index('timestamp').asfreq('D', method='ffill').reset_index()

if len(df) < 3:
    raise ValueError(f"Only {len(df)} timestamps found - need at least 3.")

# 3. Create TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
print("Successfully created TimeSeriesDataFrame:")
print(train_data.head())

# 4. Initialize predictor
predictor = TimeSeriesPredictor(
    prediction_length=30,
    path="autogluon-m4-daily-nvda",
    target="target",
    eval_metric="MASE",
    freq="D"
    # Removed known_covariates_names to avoid needing future data
)

predictor.fit(
    train_data,
    presets="high_quality",
    time_limit=1200,
    num_val_windows=3,
    hyperparameters={"TemporalFusionTransformer": {"epochs": 50}}
)

# 5. Print leaderboard
leaderboard = predictor.leaderboard(train_data)
print("Leaderboard:")
print(leaderboard)

# 6. Generate predictions
predictions = predictor.predict(train_data)
print("Forecast for NVDA (next 30 days):")
print(predictions)

# 7. Save forecast
predictions.to_csv("nvda_daily_forecast.csv")
print("Forecast saved to 'nvda_daily_forecast.csv'")

# 8. Plot historical data and forecast
plt.figure(figsize=(14, 7))
plt.plot(train_data.index.get_level_values('timestamp'), train_data['target'], label='Historical', color='blue')
forecast_dates = predictions.index.get_level_values('timestamp')
plt.plot(forecast_dates, predictions['mean'], label='Mean Forecast', color='orange', marker='o')
plt.fill_between(forecast_dates, predictions['0.1'], predictions['0.9'], alpha=0.3, color='orange', label='10th-90th Percentile')
plt.title('NVDA Price Forecast (Next 30 Days) with Historical Data')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()