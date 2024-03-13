import re
from yahoo_fin import stock_info as si
from time import sleep

def convert_market_cap_to_number(market_cap_str,):
    """
    Convert market cap string to number in dollars.
    E.g., '1.5B' -> 1500000000, '200M' -> 200000000
    """
    factors = {'T': 1e12, 'B': 1e9, 'M': 1e6, 'K': 1e3}
    factor = market_cap_str[-1]
    if factor in factors:
        return float(market_cap_str[:-1]) * factors[factor]
    else:
        
        return float(market_cap_str)
    
def extract_tickers(text, minimum_market_cap=1e10):
    ticker_pattern = r'\b[A-Za-z]{2,6}\b'
    
    potential_tickers = re.findall(ticker_pattern, text)
    
    valid_tickers = []
    for ticker in potential_tickers:
        ticker = ticker.upper()
        
        sleep(0.1)
        try:
            price = si.get_live_price(ticker)
            
            if price > 0:
                market_cap_str = si.get_stats_valuation(ticker)
                market_cap_row = market_cap_str[market_cap_str[0] == "Market Cap (intraday)"]
            if not market_cap_row.empty:
                market_cap_str = market_cap_row.iloc[0, 1]
                market_cap = convert_market_cap_to_number(market_cap_str)
                if market_cap > 1e9:  # Market cap greater than 1 billion.
                    valid_tickers.append(ticker)
                        
        except Exception as e:
            pass
            
    return valid_tickers

if __name__ == "__main__":
    text = "I think $AAPL is a great stock, but I'm not sure about $TSLA"
    print(extract_tickers(text))
