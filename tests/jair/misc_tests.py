import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers):
    # Define the fields we want to fetch
    fields = ['Name', 'Volume', 'Open', 'Close', 'High', 'Low', 'Change']

    # Create an empty DataFrame to store the results
    result_table = pd.DataFrame(columns=fields)

    for ticker in tickers:
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')

        # Calculate the percentage change
        change = (hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1] * 100

        # Create a new DataFrame for the current row
        current_row = pd.DataFrame({
            'Name': [ticker],
            'Volume': [hist['Volume'].iloc[-1]],
            'Open': [hist['Open'].iloc[-1]],
            'Close': [hist['Close'].iloc[-1]],
            'High': [hist['High'].iloc[-1]],
            'Low': [hist['Low'].iloc[-1]],
            'Change': [change]
        })

        # Append the current row to the result table
        result_table = pd.concat([result_table, current_row], ignore_index=True)

    return result_table

# Example usage
tickers = ['AAPL', 'MSFT', 'GOOGL']
stock_data = fetch_stock_data(tickers)
print(stock_data)
