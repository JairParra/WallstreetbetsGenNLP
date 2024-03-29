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

# custom utils 
from src.data_extractor import create_reddit_csv
from src.utils import load_data_from_zip

# default configurations 
LOAD_SAMPLE_DATA = False

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
        df = load_data_from_zip('data_raw/reddit_wsb.csv.zip')
        print(df)
    else: 
        # Create file temppath 
        datapath = os.path.join("data_temp", "reddit.csv")
        
        # Fetch data from the reddit API 
        create_reddit_csv(subreddit_name= "wallstreetbets", csv_name=datapath, limit=10)
    
        # Load the data from the data path
        df = pd.read_csv(datapath)

    ### Option 2: Fetch data from Reddit API

    # TODO 




