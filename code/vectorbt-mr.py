# This is a script to calculate Mean Reversion trading strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Default parameters for calculating rolling mean and standard deviation for a Mean Reversion Strategy
lookback_window = 20 # Define lookback window for calculating rolling mean and standard deviation
lower_threshold = -1 # Adjust the lower threshold for identifying oversold conditions
upper_threshold = 1 # Adjust the upper threshold for identifying overbought conditions

# Calculate rolling mean and standard deviation
rolling_mean = close_prices.rolling(window=lookback_window).mean()
rolling_std = close_prices.rolling(window=lookback_window).std()

# Calculate z-score
z_score = (close_prices- rolling_mean) / rolling_std

# Generate buy signals when z-score is below lower threshold and sell signals when z-score is above upper threshold
entries = z_score < lower_threshold
exits = z_score > upper_threshold

# Default parameters to optimize the portfolio creation process
fees = 0.001 # Transaction fees (e.g., 0.001 corresponds to 0.1% fee)
freq = 'D' # Frequency of the data (e.g., 'D' for daily, 'H' for hourly)

# Use vectorbt to simulate trades based on these signals
portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, fees=fees, freq=freq)

# Get and print the performance metrics
performance = portfolio.stats()
print(performance)

# Plot the portfolio and trades
portfolio.plot().show()