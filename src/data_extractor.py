"""
data_extractor.py
    This script contains functions for data extraction from Reddit using the corresponding API. 
    It also contains functions for data extraction from Yahoo Finance using the yfinance library.

@author: Cédric Lam
"""

################## 
### 1. Imports ###
##################

# General 
import re 
import praw 
import warnings 
from time import sleep  # sleep to pause execution between API requests
from typing import List, Dict  # Import typing for type annotations
from pandas import DataFrame 
from datetime import datetime, timedelta
# Data related 
import pandas as pd
from yahoo_fin import stock_info as si  # Yahoo_fin for fetching stock information

# Custom 
from src.utils import suppress_stdout

# Suppress warnings from the praw package
warnings.filterwarnings("ignore", module="praw")

##########################
### 2. Utils Functions ###
##########################
# Create a Reddit instance
reddit = praw.Reddit(
        client_id="YnmRgUfHOn5foh17UNLsrA",
        client_secret="EcvOf0J1NWVyuF3PTmxGkAAiuqQLkw",
        user_agent="testscript by /u/tailinks",
    )

    
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


def get_post_body(submission):
    """
    Extracts the body of a Reddit post.
    
    Parameters:
    - submission: A Reddit submission object.
    
    Returns:
    - str: The body of the Reddit post.
    """
    if submission.is_self:
        return submission.selftext
    else:
        return ""

def add_body_column(df):
    """
    Adds a new column called 'body' to the DataFrame and populates it with the body of each Reddit post.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the Reddit post data.
    
    Returns:
    - DataFrame: The modified DataFrame with the 'body' column added.
    """
    df['body'] = ""  # Initialize the 'body' column
    
    for index, row in df.iterrows():
        submission = reddit.submission(id=row['id'])
        body = get_post_body(submission)
        df.at[index, 'body'] = body
        sleep(0.2)  # Pause for 0.1 seconds
        
    return df

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
    data: DataFrame = si.get_data(ticker, start_date - timedelta(days=1), end_date, interval=interval)
    data.drop(columns=['ticker'], inplace=True)  # Drop the ticker column
    return data

def get_technical_indicators(ticker: str, date_timestamp: int,  short_ma_days = 10, long_ma_days = 50, ma_tolerance=0.005) -> Dict[str, float]:
    """
    Calculate the technical indicators of a given ticker for a specified time period ending at a given date.
    
    Parameters:
    - ticker (str): The ticker symbol.
    - date_timestamp (int): The end date for the calculation period, specified as a Unix timestamp.
    
    Returns:
    - Dict[str, float]: A dictionary containing the technical indicators for the specified period.
    
    """
    
    # Calculate the start and end dates for data fetching
    end_date = datetime.fromtimestamp(date_timestamp) - timedelta(days=1)
    start_date = end_date - timedelta(days=long_ma_days)
    
    # Fetch historical data for the ticker
    ticker_data = get_ticker_historical(ticker, start_date=start_date, end_date=end_date)
    
    # Calculate the technical indicators
    technical_indicators = {}
    
    # Calculate the moving average
    long_ma = ticker_data['adjclose'].mean()
    short_ma = ticker_data.tail(short_ma_days)['adjclose'].mean()
    tolerance = ma_tolerance  # This can be adjusted based on how sensitive you want the neutral signal to be.
    difference_percentage = (short_ma - long_ma) / long_ma
    if difference_percentage > tolerance:
        technical_indicators['moving_average'] = 'bullish'
    elif difference_percentage < -tolerance:
        technical_indicators['moving_average'] = 'bearish'
    else:
        technical_indicators['moving_average'] = 'neutral'
        
        
    
    
    # Calculate the relative strength index (RSI)
    delta = ticker_data.tail(14)['adjclose'].diff()
    gain = (delta.where(delta > 0, 0)).mean()
    loss = (-delta.where(delta < 0, 0)).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi > 70:
        technical_indicators['rsi'] = "bearish"
    if rsi < 30:
        technical_indicators['rsi'] = "bullish"
    else:
        technical_indicators['rsi'] = "neutral"
    
    
    
    # ROC Calculation
    n_periods = 10  # Number of periods to calculate ROC
    close_prices = ticker_data['adjclose']
    roc = ((close_prices.iloc[-1] - close_prices.iloc[-n_periods]) / close_prices.iloc[-n_periods]) * 100
    if roc > 0:
        technical_indicators['roc'] = "bullish"
    elif roc < 0:
        technical_indicators['roc'] = "bearish"
    else:
        technical_indicators['roc'] = "neutral"
    
    bearish_Count = 0
    bullish_Count = 0
    neutral_Count = 0

    for values in technical_indicators.values():
        if values == "bearish":
            bearish_Count += 1
        elif values == "bullish":
            bullish_Count += 1
        else:
            neutral_Count += 1
    if bearish_Count > bullish_Count and bearish_Count > neutral_Count:
        return "bearish"
    elif bullish_Count > bearish_Count and bullish_Count > neutral_Count:
        return "bullish"
    else:
        return "neutral"

#########################
### 3. Core Functions ###
#########################
def create_reddit_csv(subreddit_name= "wallstreetbets", csv_name="reddit.csv", limit=1000)-> None: 
    """
    Fetches data from the specified subreddit and saves it to a CSV file.

    Args:
        subreddit_name (str): The name of the subreddit to fetch data from. Default is "wallstreetbets".
        csv_name (str): The name of the CSV file to save the data to. Default is "reddit.csv".

    Returns:
        None
    """

    # List to store submission data
    data = []

    with suppress_stdout():
        # Fetch the top submissions from the "wallstreetbets" subreddit
        for submission in reddit.subreddit(subreddit_name).top(limit=limit):
            # Create a dictionary to store the submission data
            submission_data = {
                "title": submission.title,
                "score": submission.score,
                "id": submission.id,
                "url": submission.url,
                "comms_num": submission.num_comments,
                "created": submission.created,
            }
            # Append the submission data to the list
            data.append(submission_data)
        for submission in reddit.subreddit(subreddit_name).hot(limit=limit):
            # Create a dictionary to store the submission data
            submission_data = {
                "title": submission.title,
                "score": submission.score,
                "id": submission.id,
                "url": submission.url,
                "comms_num": submission.num_comments,
                "created": submission.created,
            }
            # Append the submission data to the list
            data.append(submission_data)
        for submission in reddit.subreddit(subreddit_name).new(limit=limit):
            # Create a dictionary to store the submission data
            submission_data = {
                "title": submission.title,
                "score": submission.score,
                "id": submission.id,
                "url": submission.url,
                "comms_num": submission.num_comments,
                "created": submission.created,
            }
            # Append the submission data to the list
            data.append(submission_data)
            
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=['id'], inplace=True)
    df = add_body_column(df)

    # Convert the 'created' column to datetime format
    df['timestamp'] = pd.to_datetime(df['created'], unit='s')

    # Export the DataFrame to a CSV file
    df.to_csv(csv_name, index=False)
    

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
    data: DataFrame = si.get_data(ticker, start_date - timedelta(days=1), end_date, interval=interval)
    data.drop(columns=['ticker'], inplace=True)  # Drop the ticker column
    return data

def get_past_average_return(tickers: List[str], date_timestamp: int, delta: str) -> float:
    """
    Calculate the average return of a list of tickers for a specified time period ending at a given date.
    
    Parameters:
    - tickers (List[str]): A list of ticker symbols.
    - date_timestamp (int): The end date for the calculation period, specified as a Unix timestamp.
    - delta (str): The time period for which the average return is calculated. Options are "Quarterly", "Monthly", "Biweekly", "Weekly".
    
    Returns:
    - float: The average return of the tickers for the specified period.
    
    Raises:
    - ValueError: If an invalid `delta` value is provided.
    """
    
    # Return 0 immediately if the list of tickers is empty
    if len(tickers) == 0:
        return 0
    
    # Define possible time deltas in days
    deltas = {"Quarterly": 91, "Monthly": 30, "Biweekly": 14, "Weekly": 7}
    
    # Validate the delta parameter
    if delta not in deltas:
        raise ValueError("Invalid delta value. Please use one of the following: Quarterly, Monthly, Biweekly, Weekly")
    
    # Calculate the start and end dates for data fetching
    end_date = datetime.fromtimestamp(date_timestamp) - timedelta(days=1)
    start_date = end_date - timedelta(days=deltas[delta])
    
    ticker_data = {}
    
    # Fetch historical data for each ticker
    for ticker in tickers:
        ticker_data[ticker] = get_ticker_historical(ticker, start_date=start_date, end_date=end_date)
    
    returns = []
    
    # Calculate the compounded return for each ticker
    for ticker in ticker_data:
        return_ticker = ticker_data[ticker]['adjclose'][-1] / ticker_data[ticker]['adjclose'][0]
        returns.append(return_ticker)
    
    # Calculate the average return across all tickers and adjust for base 1
    return sum(returns) / len(returns) - 1

def get_technical_indicators_fromlist(tickers: List[str], date_timestamp: int,  short_ma_days = 10, long_ma_days = 50, ma_tolerance=0.005):
    values = []
    for ticker in tickers:
        values.append(get_technical_indicators(ticker, date_timestamp, short_ma_days=short_ma_days, long_ma_days=long_ma_days, ma_tolerance=ma_tolerance))
    
    bearish_Count = 0
    bullish_Count = 0
    neutral_Count = 0
    for value in values:
        if value == "bearish":
            bearish_Count += 1
        elif value == "bullish":
            bullish_Count += 1
        else:
            neutral_Count += 1
    if bearish_Count > bullish_Count and bearish_Count > neutral_Count:
        return "bearish"
    elif bullish_Count > bearish_Count and bullish_Count > neutral_Count:
        return "bullish"
    else:
        return "neutral"
if __name__ == "__main__":
    create_reddit_csv()