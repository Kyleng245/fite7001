# This is a script to calculate simple moving average trading strategy
import yfinance as yf
import vectorbt as vbt

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Calculate moving averages
short_sma = close_prices.rolling(window=10).mean()
long_sma = close_prices.rolling(window=50).mean()

# Generate signals: where short SMA crosses above long SMA (entry), and below (exit)
entries = short_sma > long_sma
exits = short_sma <= long_sma

# Use vectorbt to simulate trades based on these signals
portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, fees=0.001, freq='1D')

# Get and print the performance metrics
performance = portfolio.stats()
st.write("Performance Metrics:")
st.write(performance)
print(performance)

# Plot the portfolio and trades
portfolio.plot().show()

