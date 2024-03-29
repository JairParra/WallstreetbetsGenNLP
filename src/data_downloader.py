from typing import List, Dict  
from pandas import DataFrame 
from typing import List, Dict  # Import typing for type annotations
from time import sleep
import pandas as pd
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si  # Yahoo_fin for fetching stock information

import re 

def get_ticker_historical(ticker: str, start_date: str, end_date: str, interval: str = "1d", local=False) -> DataFrame:
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
    if local:
        data_path = "temp/data/" + ticker + ".csv"
        data = pd.read_csv(data_path)
        data.index = pd.to_datetime(data.index)
        data = data.loc[start_date:end_date]
        
    else:
        data: DataFrame = si.get_data(ticker, start_date - timedelta(days=1), end_date, interval=interval)
    data.drop(columns=['ticker'], inplace=True)  # Drop the ticker column
    return data

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
            
    return list(set(valid_tickers)) # Return unique tickers

def get_all_tickers(text_list,) ->List[str]:
    """
    Extracts and returns all valid ticker symbols from a list of texts.
    
    Parameters:
    - text_list (List[str]): A list of texts to search for ticker symbols.
    
    Returns:
    - List[str]: A list of valid ticker symbols.
    """
    all_tickers: List[str] = []
    for text in text_list:
        all_tickers += extract_tickers(text)
    return list(set(all_tickers))  # Return unique tickers

def download_all(tickers,) -> None:
    for ticker in tickers:
        data = si.get_data(ticker, start_date=datetime.now() - timedelta(days=2000), end_date=datetime.now(), interval="1d")
        data.to_csv("data/temp/" + ticker + ".csv", index=True, index_label="date")
        print(f"Downloaded {ticker} data")
        
def main(text_list,) -> None:
    tickers = get_all_tickers(text_list)
    download_all(tickers)
    print("Downloaded all data")
    
# if __name__ == "__main__":
#     text_list = ["AAPL is a great company", "I think TSLA is overvalued", "AMZN has a strong balance sheet"]
#     main(text_list)