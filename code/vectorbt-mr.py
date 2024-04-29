# This is a script to calculate Mean Reversion trading strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Define lookback window for calculating rolling mean and standard deviation
lookback_window = 20

# Calculate rolling mean and standard deviation
rolling_mean = close_prices.rolling(window=lookback_window).mean()
rolling_std = close_prices.rolling(window=lookback_window).std()

# Calculate z-score
z_score = (close_prices- rolling_mean) / rolling_std

# Define z-score thresholds for mean reversion
lower_threshold = -1
upper_threshold = 1

# Generate buy signals when z-score is below lower threshold and sell signals when z-score is above upper threshold
entries = z_score < lower_threshold
exits = z_score > upper_threshold

# Use vectorbt to simulate trades based on these signals
portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, fees=0.001, freq='1D')

# Get and print the performance metrics
performance = portfolio.stats()
st.write("Performance Metrics:")
st.write(performance)
print(performance)

# Plot the portfolio and trades
portfolio.plot().show()

