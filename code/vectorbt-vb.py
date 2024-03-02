# This is a script to calculate Volatility Breakout strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Calculate rolling volatility
rolling_volatility = close_prices.pct_change().rolling(window=20).std()

# Generate buy signals when price breaks above previous high plus rolling volatility
entries = close_prices > close_prices.shift(1) + rolling_volatility
exits = close_prices < close_prices.shift(1) + rolling_volatility

# Use vectorbt to simulate trades based on these signals
portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, fees=0.001, freq='1D')

# Get and print the performance metrics
performance = portfolio.stats()
print(performance)

# Plot the portfolio and trades
portfolio.plot().show()