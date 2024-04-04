"""
main.py
    This script constitues the entrypoint to running one iteration of our project. 

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# general 
import time
import pprint 
import warnings 
from tqdm import tqdm

# data science 
import os 
import numpy as np 
import pandas as pd
from scipy.special import expit

# natural language processing 
from transformers import pipeline 

# custom general utils
from src.utils import hide_output
from src.utils import measure_time
from src.utils import load_data_from_zip

# data & text processing
from src.data_extractor import create_reddit_csv
from src.text_preprocessor import clean_lda_text
 
# topic modelling
from src.topic_modelling import train_lda_model 
from src.topic_modelling import load_lda_model
from src.topic_modelling import extract_top_words
from src.topic_modelling import assign_topic
from src.topic_modelling import create_topics_df

# stock identification 
from src.stock_processor import extract_tickers

# transformers 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#########################
### 2. Configurations ###
#########################

### Configurations

# supress all warnings from praw 
warnings.filterwarnings('ignore', module='praw')

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint

# Change number of display columns
pd.options.display.max_columns = None

# default configurations 
LOAD_SAMPLE_DATA = False
RETRAIN = True
verbose = 1


# declare base possible topic names 
# later to be loaded from a local file
candidate_topic_names = [
    "Stock Market Analysis",
    "Investment Strategies",
    "Financial Growth and Valuation",
    "Corporate Earnings and Revenue",
    "Market Trends and Predictions",
    "Social Media Trading",
    "Retail Investor Sentiment",
    "GameStop and Meme Stocks",
    "Online Trading Platforms",
    "Market Disruptions by Retail Investors",
    "Technological Advancements",
    "Emerging Industries",
    "Product Innovation",
    "Market Disruption",
    "Strategic Partnerships and Contracts",
    "Stock Trading Strategies",
    "Market Volatility",
    "Short Selling and Squeezes",
    "Options Trading",
    "Market Liquidity and Volume"
]

############### 
### 3. Main ###
###############

if __name__ == '__main__':
    
    #####################
    ### 0. Preloaders ###
    #####################
    
    # load pretrained zero-shot classification model
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        
    #######################
    ### 1. Data Loading ###
    #######################

    ### Option 1: Load Sample Datas

    if LOAD_SAMPLE_DATA:
        print("1. Loading sample data...")
        df = load_data_from_zip('data_raw/reddit_wsb.csv.zip')
        print(df) 
        
    ### Option 2: Fetch data for run using the API 
    else: 
        print("*"*100 + "\n1. Fetching Reddit data...\n" + "*"*100)
        
        # Create file temppath 
        datapath = os.path.join("data_temp", "reddit.csv")
        
        # Fetch data from the reddit API 
        _ = measure_time(
            hide_output(
                 create_reddit_csv(subreddit_name="wallstreetbets", 
                                   csv_name=datapath, limit=10)                   
            )
        )
    
        # Load the data from the data path
        df = pd.read_csv(datapath)
        print(df)
        

    ##########################
    ### 2. Topic Modelling ###
    ##########################
    
    # Extract all the titles from the dataframe
    if LOAD_SAMPLE_DATA: 
        texts = df['text'].tolist().iloc[0:30]
    else: 
        texts = df['text'].tolist()  
    
    # Clean the corpus for this iteration
    clean_texts = clean_lda_text(texts, clean_emojis=True, verbose=True)
    
    if RETRAIN: 
        # Create a dictionary with the parameters used in the LDA model 
        lda_params = {
            'num_topics': 4,                 # The number of requested latent topics to be extracted from the training corpus
            'update_every': 1,               # Number of documents to be iteratively updated
            'chunksize': 100,                 # Number of documents to be used in each training chunk
            'passes': 7,                     # Number of passes through the corpus during training
            'alpha': 'symmetric',            # Hyperparameter affecting sparsity/thickness of the topics
            'iterations': 100,               # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
        }
        
        # Train the LDA model
        lda_model, corpus, id2word = train_lda_model(clean_texts[0:100], lda_params=lda_params)
        
        # Display the topics
        pprint(lda_model.print_topics())

        # Add an index column to the dataframe
        df['index'] = range(len(df))
        
    else: 
        
        # Example usage
        model_dir = "models/lda_model"
        lda_model, id2word = load_lda_model(model_dir)
        
        # Reconstruct corpus from clean_texts
        corpus = [id2word.doc2bow(text) for text in clean_texts]
        
        
    # Extract top n words per topic
    top_words_dict = extract_top_words(lda_model, top_n=40)
    
    # initialize topic names dict
    topic_names = {}
    
    # Assign a topic name for each of the topics 
    for topic_id, top_words in tqdm(top_words_dict.items(), desc="Labeling topics..."):
        
        # Extract topic tokens into a single string 
        toks_str = ' '.join(top_words)
        
        # assign the most likely label to these topics
        topic_names[topic_id] = classifier(toks_str, 
                                           candidate_topic_names, 
                                           multi_label=False)['labels'][0]

    # Print the topics and their top words
    if verbose >= 1:
        for topic_id, topic_label in topic_names.items():
            print(f"Topic {topic_id}: {topic_label}")
            print(f"First 5 words: {', '.join(top_words_dict[topic_id][:5])}")
            print()
    
    # Assign topics to the texts
    df_assigned_topics = create_topics_df(df['text'], lda_model, 
                                          topic_names).drop(columns=["doc_text"])
    
    # Left join the assigned topics to the original dataframe
    df_join = df.merge(df_assigned_topics, on='index')
    
    #############################
    ### 3. Sentiment Analysis ###
    #############################
    
    
    #########################
    ### 4. Trend Analysis ###
    #########################
    
    
    #########################
    ### 5. Stock Analysis ###
    #########################
    
    # extract the tickers from all the texts in the df_join and store them in a new column
    start_time = time.time()

    df_join['tickers'] = df_join['text'].apply(extract_tickers)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    







