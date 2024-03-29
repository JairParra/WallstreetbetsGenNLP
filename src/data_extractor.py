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
import praw 
import warnings 
from time import sleep  # sleep to pause execution between API requests

# Data related 
import pandas as pd

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
    

if __name__ == "__main__":
    create_reddit_csv()