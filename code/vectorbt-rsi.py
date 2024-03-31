# This is a script to calculate Relative Strength Index strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Default Relative Strength Index parameters
window = 14 # Adjust the window size for calculating RSI
rsi_buy_threshold = 30 # RSI oversold threshold
rsi_sell_threshold = 70 # RSI overbought threshold

# Calculate Relative Strength Index
rsi = vbt.RSI.run(close_prices, window=window)

# Plot with Relative Strength Index on price
kwargs1 = {"title_text" : "Relative Strength Index on Price", "title_font_size" : 15}
rsi.plot(**kwargs1).show()

# Generate buy signals when RSI is below oversold threshold and sell signals when RSI is above overbought threshold
entries = rsi.rsi_below(rsi_buy_threshold)
exits = rsi.rsi_above(rsi_sell_threshold)

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