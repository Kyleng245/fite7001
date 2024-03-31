# This is a script to calculate Bollinger Bands trading strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Default parameters of the moving average used to calculate the Bollinger Bands
window = 20 # Length of the moving average window
ewm = False # Center of the bands: False indicates simple moving average (SMA) used; True indicates exponential moving average (EMA) used
alpha = 2 # Specify the smoothing factor of exponential moving average (EMA) when ewm is set to True

# Calculate Bollinger Bands
bbands = vbt.BBANDS.run(close_prices, window=window, ewm=ewm, alpha=alpha)

# Plot with Bollinger Bands on price
kwargs1 = {"title_text" : "Bollinger Bands on Price", "title_font_size" : 15}
bbands.plot(**kwargs1).show()

# Generate buy signals when price crosses below lower band and sell signals when price crosses above upper band
entries = close_prices < bbands.lower
exits = close_prices > bbands.upper

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