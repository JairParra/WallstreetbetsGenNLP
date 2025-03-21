"""
utils.py
    This script contains miscellaneous utility functions.

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# general
import os 
import sys
import time
import zipfile 
import pandas as pd
from tqdm import tqdm
from collections import Counter
from contextlib import contextmanager

# visualization 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

########################
### 2. General Utils ###
########################

def load_data_from_zip(zip_file_path:str = 'data_raw/reddit_wsb.csv.zip') -> pd.DataFrame:
    """
    Loads the reddit data from a zip file and returns it as a DataFrame. 
    
    Args:
    file_path (str): The file path to load the data from.
    
    Returns:
    pd.DataFrame: The loaded data.
    """

    # Partition the path into the directory and the file name
    directory, file_name = os.path.split(zip_file_path)

    # retrieve the filename with the zip extension removed
    file_name_no_ext = os.path.splitext(file_name)[0]

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents of zip file in current directory 
        zip_ref.extractall(directory)

    # Read the CSV using Pandas
    csv_file_path = os.path.join(directory, file_name_no_ext)
    df = pd.read_csv(csv_file_path)

    # Fill all the NaN values in the body column with an empty string
    df['body'] = df['body'].fillna('')

    # Combine the title and bodyy into a single column text, separated by two newlines
    df['text'] = df['title'] + '\n\n' + df['body']

    # drop the body column 
    df = df.drop(columns=['body'])

    return df

@contextmanager
def suppress_output():
    """
    A context manager that suppresses all output to IO and printing messages.
    """
    # Open a null file as the output stream
    with open(os.devnull, 'w') as devnull:
        # Replace the standard output and error streams with the null file
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            # Yield control back to the caller
            yield
        finally:
            # Restore the standard output and error streams
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

def hide_output(func):
    """
    A decorator function that hides all output to IO and printing messages from a given function.

    Parameters:
    - func: The function to be decorated.

    Returns:
    - result: The result of the function.

    Usage:
    - @hide_output
        def my_function():
            # code here
            return result
    """
    def wrapper(*args, **kwargs):
        with suppress_output():
            result = func(*args, **kwargs)
        return result
    return wrapper

def measure_time(func):
    """
    A decorator function that measures the execution time of a given function.

    Parameters:
    - func: The function to be measured.

    Returns:
    - result: The result of the function.

    Usage:
    - @measure_time
      def my_function():
          # code here
          return result
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return result
    return wrapper

##########################
### 3. Sentiment Utils ###
##########################

def extract_text(df):
    """
    This function takes a DataFrame and concatenates the 'title' and 'body' columns into a new 'text' column.
    The 'body' column is first filled with empty strings for NaN values. The 'text' column is created by concatenating
    the 'title' and 'body' columns, separated by two newlines. Finally, the 'body' column is dropped from the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing at least 'title' and 'body' columns.
    
    Returns:
    - df: DataFrame with the 'body' column removed and 'text' column added.
    - texts: New DataFrame containing only the 'text' column.
    
    Usage:
    - Assuming 'df' is your DataFrame
    -   df, texts = concat_text(df)
        print(df.head())  # To preview the modified DataFrame
        print(texts.head())  # To preview the new texts DataFrame
    """
    # Fill all the NaN values in the body column with an empty string
    df['body'] = df['body'].fillna('')
    
    # Combine the title and body into a single column text, separated by two newlines
    df['text'] = df['title'] + '\n\n' + df['body']
    
    # Drop the body column
    df = df.drop(columns=['body'])
    
    # Create a new DataFrame containing only the 'text' column
    texts = pd.DataFrame(df['text'])
    
    # Return both the modified original DataFrame and the new texts DataFrame
    return df, texts


def load_words_from_csv(file_path):
    """
    Load words from a CSV file into a Python list.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - list: A list of words loaded from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df['word'].tolist()