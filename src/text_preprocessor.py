
"""
toic_modelling.py
    This script contains functions relating to topic modelling and LDA analysis.

@author: Hair Parra
@author: Jonathan Gonzales
"""

################## 
### 1. Imports ###
##################

# General imports
import os 
import pprint 
import zipfile 
import logging
from collections import Counter
from typing import Union, List, Tuple, Dict, Any
from tqdm import tqdm 

# Data Analysis and visualizations
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# NLTK setup
import nltk 
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Text Processing 
import re 
import spacy
import gensim
from gensim import corpora
from emoji import demojize
from spacy.tokens import Doc
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim.models.callbacks import PerplexityMetric
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

# Dedicated NLP Visualizations 
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud

# Custom scripts 
from src.utils import format_topics_sentences
from src.utils import plot_topic_keywords

# Configurations 
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Exclude common negation words from the stop words list
# negation_words = {'no', 'not', 'nor', 'neither', 'never', "n't", 'none', 'through'}
except_words = {'through'}
stop_words = stop_words - except_words

# Load Spacy model and disable irrelevant components for acceleration
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
# nlp.max_length = 1500000  # Adjust based on your text size

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint

# Ignore warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


###################### 
### 2. Local Utils ###
######################

# Utils used for multiple functions in this script


#################################
### 3. LDA Text Preprocessing ###
#################################

def clean_lda_text(texts: Union[str, List[str], pd.Series], clean_emojis: bool = False, verbose:bool = False) -> Union[str, List[str]]:
    """
    Clean and preprocess text data for topic modeling.

    This function performs several preprocessing steps on a list of text documents:
    - Removes hyperlinks.
    - Tokenizes and lemmatizes the text using spaCy.
    - Converts and cleans emojis.
    - Removes non-alphabetic characters, stopwords, and short tokens.
    - Filters out verbs to reduce noise.
    - Forms bigrams and trigrams, filtering out those containing stopwords or non-alphabetic characters.
    - Splits n-grams longer than 4 into individual tokens.

    Parameters:
    - texts: A list of text documents to be cleaned.
    - clean_emojis: A boolean indicating whether emojis should be removed from the text.

    Returns:
    - A list of cleaned text documents, with each document represented as a list of tokens.
    """

    # Preprocess the texts to remove hyperlinks
    texts = [re.sub(r'http\S+|www\S+|ftp\S+', '', text) for text in texts]

    # Define a function to clean a single document
    def clean_doc(doc: Doc, clean_emojis: bool = False) -> List[str]:
        # Filter out tokens that are stopwords, verbs, or have length less than 2, then lemmatize and lowercase
        tokens = [demojize(token.lemma_ if token.lemma_ != '-PRON-' else token.text).lower() for token in doc
                  if not (token.is_stop or token.pos_ == 'VERB' or len(token.text) < 2)]

        # Convert and clean emojis
        tokens = [re.sub(r':', '_', token) if token.startswith(':') and token.endswith(':') else token for token in tokens]
        if clean_emojis:
            tokens = [re.sub(r'_.*_', '', token) for token in tokens]

        # Remove non-alphabetic characters except for '_'
        tokens = [re.sub(r'[^a-z_]', '', token) for token in tokens]

        return tokens

    # Apply the cleaning function to each document
    cleaned_texts = [clean_doc(doc, clean_emojis) for doc in tqdm(nlp.pipe(texts, batch_size=50), 
                                                                  total=len(texts), 
                                                                  desc="Cleaning Texts",
                                                                  disable=not verbose)]

    # Form bigrams and trigrams
    bigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(cleaned_texts, min_count=3, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS))
    trigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(bigram_mod[cleaned_texts], threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS))
    cleaned_texts = [bigram_mod[doc] for doc in tqdm(cleaned_texts, desc="Creating bigrams...", disable=not verbose)]
    cleaned_texts = [trigram_mod[bigram_mod[doc]] for doc in tqdm(cleaned_texts, desc="Creating trigrams...", disable=not verbose)]

    # Filter out bigrams and trigrams containing stopwords or non-alphabetic characters, and split n-grams longer than 4
    cleaned_texts = [[token for sub_token in doc for token in 
                      (sub_token.split('_') if len(sub_token.split('_')) > 4 else [sub_token])
                      if all(word not in stop_words and re.match('^[a-z_]+$', word) for word in token.split('_'))]
                     for doc in cleaned_texts]

    return cleaned_texts


#######################################
### 4. Sentiment Text Preprocessing ###
####################################### 

def clean_sentiment_text(raw_text: str) -> str:
    """
    Clean and preprocess text data for sentiment classification.

    This function performs text cleaning steps on a raw text document:
    - Step1 
    - Step2

    Parameters:
    - raw_text: The raw text document to be cleaned.

    Returns:
    - The cleaned text document.
    """

    # TODO: Implement text cleaning for sentiment analysis
    text_clean = raw_text
    
    return text_clean