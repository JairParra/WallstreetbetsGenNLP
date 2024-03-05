import re
from yahoo_fin import stock_info as si
from time import sleep

def extract_and_validate_stock_tickers(text):
    ticker_pattern = r'\b[A-Za-z]{2,6}\b'
    
    potential_tickers = re.findall(ticker_pattern, text)
    
    valid_tickers = []
    for ticker in potential_tickers:
        ticker = ticker.upper()
        sleep(0.1)
        try:
            price = si.get_live_price(ticker)
            if price > 0:
                valid_tickers.append(ticker)
        except Exception as e:
            pass
            
    return valid_tickers

# Example usage:
text = "Investing in the stock market is popular, with companies like AAPL, MSFT, GOOGL, and Gme being prominent players."
extracted_and_validated_tickers = extract_and_validate_stock_tickers(text)
print(extracted_and_validated_tickers)
