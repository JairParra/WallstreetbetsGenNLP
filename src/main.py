"""
main.py
    This script constitues the entrypoint to running one iteration of our project. 

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# general 
import warnings 

# data science 
import os 
import pandas as pd

# natural language processing 
import nltk 

# custom utils 
from src.utils import load_data_from_zip
from src.data_extractor import create_reddit_csv
from src.text_preprocessor import clean_lda_text 

# default configurations 
LOAD_SAMPLE_DATA = False
RETRAIN = False

# supress all warnings from praw 
warnings.filterwarnings('ignore', module='praw')

############### 
### 2. Main ###
###############

if __name__ == '__main__':
    
    #######################
    ### 1. Data Loading ###
    #######################

    ### Option 1: Load Sample Datas

    if LOAD_SAMPLE_DATA:
        print("1. Loading sample data...")
        df_test = load_data_from_zip('data_raw/reddit_wsb.csv.zip').iloc[0:10]
        print(df_test) 
        
    ### Option 2: Fetch data for run using the API 
    else: 
        # Create file temppath 
        datapath = os.path.join("data_temp", "reddit.csv")
        
        # Fetch data from the reddit API 
        create_reddit_csv(subreddit_name= "wallstreetbets", csv_name=datapath, limit=10)
    
        # Load the data from the data path
        df = pd.read_csv(datapath)
        

    ##########################
    ### 2. Topic Modelling ###
    ##########################
    
    # Extract all the titles from the dataframe
    texts = df['text'].tolist()  
    
    # Clean the corpus for this iteration
    clean_texts = clean_lda_text(texts, clean_emojis=True, verbose=True)
    
    
    





