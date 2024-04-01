# This is a script to calculate Simple Moving Average (SMA) trading strategy
import yfinance as yf
import vectorbt as vbt

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01') # Adjust the historical data start time and end time interval

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Adjust parameters for calculating moving averages
short_window = 10  # Adjust the window size for the short moving average
long_window = 50  # Adjust the window size for the long moving average

# Calculate moving averages
short_sma = close_prices.rolling(window=short_window).mean()
long_sma = close_prices.rolling(window=long_window).mean()

# Generate signals: where short SMA crosses above long SMA (entry), and below (exit)
entries = short_sma > long_sma
exits = short_sma <= long_sma

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