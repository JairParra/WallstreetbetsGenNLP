import re  # Regular expressions library for matching ticker patterns
from yahoo_fin import stock_info as si  # Yahoo_fin for fetching stock information
from time import sleep  # sleep to pause execution between API requests
from typing import List, Dict, Any  # Import typing for type annotations
from pandas import DataFrame  # Import DataFrame for typing return values

def convert_market_cap_to_number(market_cap_str: str) -> float:
    """
    Convert a market capitalization string to a numeric value in dollars.
    
    Parameters:
    - market_cap_str (str): The market capitalization string (e.g., '1.5B').
    
    Returns:
    - float: The market capitalization in dollars.
    """
    factors: Dict[str, float] = {'T': 1e12, 'B': 1e9, 'M': 1e6, 'K': 1e3}
    factor: str = market_cap_str[-1]
    if factor in factors:
        return float(market_cap_str[:-1]) * factors[factor]
    else:
        return float(market_cap_str)
    
def extract_tickers(text: str, minimum_market_cap: float = 1e10) -> List[str]:
    """
    Extracts and returns valid ticker symbols from a given text based on live price and market capitalization.
    
    Parameters:
    - text (str): The text to search for ticker symbols.
    - minimum_market_cap (float, optional): The minimum market capitalization to consider a ticker valid. Defaults to 10 billion.
    
    Returns:
    - List[str]: A list of valid ticker symbols.
    """
    ticker_pattern: str = r'\b[A-Za-z]{2,6}\b'
    potential_tickers: List[str] = re.findall(ticker_pattern, text)
    valid_tickers: List[str] = []
    
    for ticker in potential_tickers:
        ticker = ticker.upper()
        sleep(0.1)  # Pause to limit the frequency of API requests
        try:
            price: float = si.get_live_price(ticker)
            if price > 0:
                market_cap_str: DataFrame = si.get_stats_valuation(ticker)
                market_cap_row: DataFrame = market_cap_str[market_cap_str[0] == "Market Cap (intraday)"]
                if not market_cap_row.empty:
                    market_cap_str: str = market_cap_row.iloc[0, 1]
                    market_cap: float = convert_market_cap_to_number(market_cap_str)
                    if market_cap >= minimum_market_cap:  # Ensure market cap meets minimum requirement
                        valid_tickers.append(ticker)
        except Exception as e:
            pass
            
    return valid_tickers

def get_ticker_historical(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> DataFrame:
    """
    Fetches historical data for a given ticker within the specified date range.
    
    Parameters:
    - ticker (str): The ticker symbol.
    - start_date (str): Start date in format YYYY-MM-DD.
    - end_date (str): End date in format YYYY-MM-DD.
    - interval (str, optional): Data interval (e.g., "1d" for daily). Defaults to "1d".
    
    Returns:
    - DataFrame: Historical data for the given ticker.
    """
    data: DataFrame = si.get_data(ticker, start_date, end_date, interval=interval)
    data['return'] = data['adjclose'].pct_change()  # Calculate returns
    data.drop(columns=['ticker'], inplace=True)  # Drop the ticker column
    return data

if __name__ == "__main__":
    # Example usage
    text: str = "I think AAPL is a great stock, but I'm not sure about $TSLA, walmart"
    tickers_list: List[str] = extract_tickers(text)
    for ticker in tickers_list:
        print(ticker)
        print(get_ticker_historical(ticker, "2019-01-01", "2025-01-01"))
