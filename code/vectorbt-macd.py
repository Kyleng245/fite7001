# This is a script to calculate Moving Average Convergence Divergence trading strategy
import vectorbt as vbt
import yfinance as yf
        # Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Calculate MACD
macd = vbt.MACD.run(close_prices)

# Plot with Moving Average Convergence Divergence on price
kwargs1 = {"title_text" : "Moving Average Convergence Divergence on Price", "title_font_size" : 13}
macd .plot(**kwargs1).show()

# Generate buy signals when MACD histogram is positive and sell signals when MACD histogram is negative
entries = macd.macd_above(0)
exits = macd.macd_below(0)

# Use vectorbt to simulate trades based on these signals
portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, fees=0.001, freq='1D')

# Get and print the performance metrics
performance = portfolio.stats()
st.write("Performance Metrics:")
st.write(performance)
print(performance)

# Plot the portfolio and trades
portfolio.plot().show()


