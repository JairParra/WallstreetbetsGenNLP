import re

def extract_stock_tickers(text):
    # Regular expression to match potential stock tickers:
    # This regex looks for words that are entirely uppercase and 1-5 characters long, optionally followed by a dot and digits (e.g., BRK.A)
    # Adjust the regex as needed based on the specific patterns of stock tickers you're interested in.
    ticker_pattern = r'\b[A-Z]{1,5}(?:\.[A-Z0-9]{1,2})?\b'
    
    # Find all matches in the text
    potential_tickers = re.findall(ticker_pattern, text)
    
    # Optionally, filter or validate tickers here (e.g., by checking them against a financial API)
    
    return potential_tickers

