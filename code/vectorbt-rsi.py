# This is a script to calculate Relative Strength Index strategy
import vectorbt as vbt
import yfinance as yf

# Fetch historical data from Yahoo Finance
ticker_symbol = 'AAPL'  # Apple Inc.
data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

# Select only the 'Close' column for simplicity
close_prices = data['Close']

# Define Relative Strength Index parameters
rsi_period = 14
rsi_buy_threshold = 30
rsi_sell_threshold = 70

# Calculate Relative Strength Index
rsi = vbt.RSI.run(close_prices, window=rsi_period)

# Plot with Relative Strength Index on price
kwargs1 = {"title_text" : "Relative Strength Index on Price", "title_font_size" : 15}
rsi.plot(**kwargs1).show()

# Generate buy signals when RSI is below oversold threshold and sell signals when RSI is above overbought threshold
entries = rsi.rsi_below(rsi_buy_threshold)
exits = rsi.rsi_above(rsi_sell_threshold)

# Use vectorbt to simulate trades based on these signals
portfolio = vbt.Portfolio.from_signals(close_prices, entries, exits, fees=0.001, freq='1D')

# Get and print the performance metrics
performance = portfolio.stats()
st.write("Performance Metrics:")
st.write(performance)
print(performance)

# Plot the portfolio and trades
st.write("Charts:")
st.plotly_chart(rsi.plot(**kwargs1))
st.plotly_chart(portfolio.plot())
