"""
sentiment_analysis.py
    This script contains functions relating to sentiment analysis. 

@author: Jonathan Gonzalez
@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# General Imports

import re
import zipfile 
from tqdm.notebook import tqdm
from emoji import demojize
from typing import Union, List

# Data Analysis and visualizations
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Import Spacy
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS as stop_words  # Import default English stop words

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Import NLTK
import nltk
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nltk.download('punkt', quiet=True)
nltk.download('sentiwordnet')
nltk.download('wordnet')

##########################
### 2. Utils Functions ###
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


#########################
### 3. Core Functions ###
#########################

def analyze_sentiment_vader(text_row):
    """
    Analyzes the sentiment of a given text using the VADER sentiment analysis tool.
    This function expects a tuple (text_row) containing an identifier (index) and a text string.
    It uses the VADER SentimentIntensityAnalyzer to compute the compound sentiment score of the text,
    which is then used to classify the sentiment into one of three categories: 'positive', 'negative', or 'neutral'.
    
    The sentiment classification is based on the compound score:
    - A compound score >= 0.05 is classified as 'positive'.
    - A compound score <= -0.05 is classified as 'negative'.
    - A compound score between -0.05 and 0.05 is classified as 'neutral'.
    
    Parameters:
    - text_row (tuple): A tuple containing an index (or any identifier) and a text string to be analyzed.
    
    Returns:
    - tuple: A tuple containing the original index (or identifier) and a tuple with the compound sentiment score
      and the sentiment label ('positive', 'negative', or 'neutral').
    
    This function is particularly useful for batch processing of texts, where each text is associated with an identifier,
    allowing for easy mapping of results back to the original dataset.
    
    """
    idx, text = text_row  # Unpack the tuple received
    score = sia.polarity_scores(text)['compound']  # Compute the compound score
    # Determine the sentiment label based on the compound score
    if score >= 0.05:
        label = 'positive'
    elif score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return idx, (score, label)

def process_with_vader(dataframe):
    """
    This function processes a pandas DataFrame to analyze the sentiment of text data in parallel using the VADER sentiment analysis tool. 
    It expects the DataFrame to have a column named 'processed_text' containing the texts to be analyzed. The function performs sentiment analysis on each text entry,
    assigning a sentiment score and label (positive, negative, or neutral) based on the compound score calculated by the VADER tool.

    The sentiment analysis is executed in parallel, utilizing a number of worker threads equal to the number of CPU cores available on the machine minus one, to optimize performance without overloading the system.

    The results of the sentiment analysis are then aggregated into two dictionaries, one mapping text indices to sentiment scores and the other to sentiment labels. These dictionaries are used to map the sentiment scores and labels back to the original DataFrame, adding two new columns: 'sentiment_score' and 'sentiment_label'.

    Parameters:
    - dataframe (pandas.DataFrame): A DataFrame containing a column 'processed_text' with the text to be analyzed.

    Returns:
    - pandas.DataFrame: The original DataFrame augmented with two new columns: 'sentiment_score' and 'sentiment_label', corresponding to the sentiment analysis results.

    This function streamlines the sentiment analysis of large datasets, allowing for efficient and scalable sentiment analysis with VADER.
    
    Usage:
    - df_with_sentiment = process_with_vader(df)
    """
    # Extract (index, text) tuples for parallel processing
    texts_to_analyze = list(dataframe['processed_text'].items())
    
    num_cores = os.cpu_count()
    max_workers = num_cores - 1  # Adjust based on your machine
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Execute analysis in parallel
        results = list(executor.map(analyze_sentiment_vader, texts_to_analyze))
    
    # Convert results into two dictionaries {index: sentiment_score} and {index: sentiment_label}
    scores = {idx: result[0] for idx, result in results}
    labels = {idx: result[1] for idx, result in results}
    
    # Map the scores and labels back to the DataFrame using the index
    dataframe['sentiment_score'] = dataframe.index.map(scores)
    dataframe['sentiment_label'] = dataframe.index.map(labels)
    return dataframe

