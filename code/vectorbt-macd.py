# This is a script to calculate Moving Average Convergence Divergence (MACD) trading strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01') # Adjust the historical data start time and end time interval

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Default parameters for calculating MACD
fast_window = 12  # Adjust the window size for the fast moving average
slow_window = 26  # Adjust the window size for the slow moving average
signal_window = 9  # Adjust the window size for the signal line
adjust = True # True indicates MACD values are adjusted to account for changes in the price of the underlying asset
macd_above_threshold = 0  # Adjust the threshold for MACD values above which buy signals are generated
macd_below_threshold = 0  # Adjust the threshold for MACD values below which sell signals are generated

# Calculate MACD
macd = vbt.MACD.run(close_prices, fast_window=fast_window, slow_window=slow_window, signal_window=signal_window, adjust=adjust)

# Plot with Moving Average Convergence Divergence on price
kwargs1 = {"title_text" : "Moving Average Convergence Divergence on Price", "title_font_size" : 13}
macd .plot(**kwargs1).show()

# Generate buy signals when MACD histogram is positive and sell signals when MACD histogram is negative
entries = macd.macd_above(macd_above_threshold)
exits = macd.macd_below(macd_below_threshold)

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