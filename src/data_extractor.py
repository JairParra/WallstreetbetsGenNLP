"""
data_extractor.py
    This script contains functions for data extraction from Reddit using the corresponding API. 
    It also contains functions for data extraction from Yahoo Finance using the yfinance library.

@author: Cédric Lam
"""

################## 
### 1. Imports ###
##################
import praw
import pandas as pd
import re  # Regular expressions library for matching ticker patterns
from yahoo_fin import stock_info as si  # Yahoo_fin for fetching stock information
from time import sleep  # sleep to pause execution between API requests
from typing import List, Dict  # Import typing for type annotations


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
                market_cap_str: pd.DataFrame = si.get_stats_valuation(ticker)
                market_cap_row: pd.DataFrame = market_cap_str[market_cap_str[0] == "Market Cap (intraday)"]
                if not market_cap_row.empty:
                    market_cap_str: str = market_cap_row.iloc[0, 1]
                    market_cap: float = convert_market_cap_to_number(market_cap_str)
                    if market_cap >= minimum_market_cap:  # Ensure market cap meets minimum requirement
                        valid_tickers.append(ticker)
        except Exception as e:
            pass
            
    return valid_tickers

def get_ticker_historical(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
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
    data: pd.DataFrame = si.get_data(ticker, start_date, end_date, interval=interval)
    data['return'] = data['adjclose'].pct_change()  # Calculate returns
    data.drop(columns=['ticker'], inplace=True)  # Drop the ticker column
    return data

if __name__ == "__main__":
    create_reddit_csv()