# This is a script to calculate Volatility Breakout strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Default parameters for calculating rolling volatility
window = 20  # Adjust the window for calculating rolling volatility
threshold_multiplier = 2  # Adjust the multiplier for defining the threshold for volatility breakout

# Calculate rolling volatility
rolling_volatility = close_prices.pct_change().rolling(window=window).std()

# Generate buy signals when price breaks above previous high plus rolling volatility
entries = close_prices > close_prices.shift(1) + threshold_multiplier * rolling_volatility
exits = close_prices < close_prices.shift(1) + threshold_multiplier * rolling_volatility

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