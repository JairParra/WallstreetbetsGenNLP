"""
sentiment_analysis.py
    This script contains functions relating to trend analysis. 

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

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
# Import NLTK
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nltk.download('punkt', quiet=True)
nltk.download('sentiwordnet')
nltk.download('wordnet')


##########################
### 2. Utils Functions ###
##########################
#used same utils as sentiment, this is the only different function
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
# Usage
bullish_words = load_words_from_csv('data_raw/bearish.csv') #are redifining 
bearish_words = load_words_from_csv('data_raw/bullish.csv')

#########################
### 3. Core Functions ###
#########################


def analyze_emotion(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence.lower())
    
    # Count bullish and bearish words
    bullish_count = sum(word in bullish_words for word in words)
    bearish_count = sum(word in bearish_words for word in words)
    
    # Determine sentiment
    if bullish_count > bearish_count:
        return "Bullish"
    elif bearish_count > bullish_count:
        return "Bearish"
    else:
        return "Neutral"

