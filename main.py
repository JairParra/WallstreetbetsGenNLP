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
import logging 
import argparse
import warnings 
import datetime
from tqdm import tqdm

# data science 
import os 
import nltk
import numpy as np 
import pandas as pd
from scipy.special import expit

# natural language processing 
from transformers import pipeline 

# custom general utils
from src.utils import hide_output
from src.utils import measure_time
from src.utils import extract_text
from src.utils import load_data_from_zip
from src.utils import load_words_from_csv

# data & text processing
from src.data_extractor import create_reddit_csv
from src.text_preprocessor import clean_lda_text
from src.text_preprocessor import clean_sentiment
 
# topic modelling
from src.topic_modelling import train_lda_model 
from src.topic_modelling import load_lda_model
from src.topic_modelling import extract_top_words
from src.topic_modelling import assign_topic
from src.topic_modelling import create_topics_df

# sentiment and trend analysis 
from src.trend_analysis import analyze_emotion
from src.sentiment_analysis import process_with_vader
from src.sentiment_analysis import analyze_sentiment_vader

# stock identification 
from src.stock_processor import extract_tickers

# transformers 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


###################
### 2. Argument ###
###################

# default configurations 
LOAD_SAMPLE_DATA = False
RETRAIN = True
LIMIT_FETCH = 10
verbose = 1
OUTPUT_PATH = "data_clean/wsb_clean.csv"

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Modify default configurations')

# Add arguments for each configuration option
parser.add_argument('--load_sample_data', action='store_true', default=False, help='Load sample data')
parser.add_argument('--retrain', action='store_true', default=True, help='Retrain the model')
parser.add_argument('--limit_fetch', type=int, default=100, help='Limit for fetching data')
parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
parser.add_argument('--custom_output_name', type=str, default=None, help='Custom name for the output file')
parser.add_argument('--start', type=int, default=0, help='Start index for sample data. Default=3000')
parser.add_argument('--end', type=int, default=3000, help='End index for end data. Default=3000')

# Parse the command line arguments
args = parser.parse_args()

# Update the default configurations with the command line arguments
LOAD_SAMPLE_DATA = args.load_sample_data
RETRAIN = args.retrain
LIMIT_FETCH = args.limit_fetch
verbose = args.verbose
OUTPUT_PATH = f"data_clean/{args.custom_output_name}.csv" if args.custom_output_name is not None else OUTPUT_PATH

# Verify that the inputs are valid
if args.start < 0 or args.end < 0 or args.start > args.end:
    raise ValueError("Invalid start and end values. Start must be less than or equal to end.")

# Assign the input values to the variables START and END
START = args.start
END = args.end

#########################
### 3. Configurations ###
#########################

### Configurations

# Download necessary NLTK data
nltk.download('corpus', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt')
nltk.download('sentiwordnet', quiet=True)
nltk.download('wordnet', quiet=True)

# supress all warnings from praw 
warnings.filterwarnings('ignore', module='praw')

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint

# Change number of display columns
pd.options.display.max_columns = None

# Configure logging
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"logs/example_{current_datetime}.log"
logging.basicConfig(filename=filename, level=logging.INFO)

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
### 4. Main ###
###############

if __name__ == '__main__':
    
    #####################
    ### 0. Preloaders ###
    #####################
    
    # start timing 
    t0 = time.time()

    # add start message to logging with specific time 
    logging.info(f"Starting the process at {datetime.datetime.now()}")
    
    # load pretrained zero-shot classification model
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    
    # Preload trend word lists
    bullish_words = load_words_from_csv("data_raw/bullish.csv") # Bullish word list
    bearish_words = load_words_from_csv("data_raw/bearish.csv") # Bearish word list
        
    #######################
    ### 1. Data Loading ###
    #######################

    ### Option 1: Load Sample Datas

    # Log start of data loading 
    logging.info("Starting data loading...")

    if LOAD_SAMPLE_DATA:
        print("1. Loading sample data...")
        df = load_data_from_zip('data_raw/reddit_wsb.csv.zip').iloc[START:END]
        print(df) 
        
    ### Option 2: Fetch data for run using the API 
    else: 
        print("#"*100 + "\n1. Fetching Reddit data...\n" + "#"*100)
        
        # Create file temppath 
        datapath = os.path.join("data_temp", "reddit.csv")
        
        # Fetch data from the reddit API 
        _ = measure_time(
            create_reddit_csv(subreddit_name="wallstreetbets", 
                              csv_name=datapath, 
                              limit=LIMIT_FETCH)                   
        )
    
        # Load the data from the data path
        df = pd.read_csv(datapath)
        print(df)

        # Log the number of rows fetched
        logging.info(f"Fetched {len(df)} rows of data.")
        
    logging.info("Data loaded successfully.")

    ##########################
    ### 2. Topic Modelling ###
    ##########################
    
    print("#"*100 + "\n2. Performing Topic Modelling...\n" + "#"*100)
    logging.info("Starting topic modelling...")

    # Extract all the titles from the dataframe
    if LOAD_SAMPLE_DATA: 
        texts = df['text'].tolist()
    else: 
        texts = df['text'].tolist()  
    
    # Clean the corpus for this iteration
    clean_texts = clean_lda_text(texts, clean_emojis=True, verbose=True)
    
    # no longer needed
    del(texts)
    
    if RETRAIN: 
        print("Retraining the model...")
        logging.info("Retraining the LDA model...")

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
        # Load the LDA model and the id2word dictionary
        print("Loading the model...")
        logging.info("Loading the LDA model...")
        
        # Example usage
        model_dir = "models/lda_model"
        lda_model, id2word = load_lda_model(model_dir)
        
        # Reconstruct corpus from clean_texts
        corpus = [id2word.doc2bow(text) for text in clean_texts]
        
    # log the completion of the topic modelling
    logging.info("Topic modelling completed.")

    # Start topic assignment section 
    print("#"*100 + "\nAssigning topics to the texts...\n" + "#"*100)
    logging.info("Assigning topics to the texts...")

    # Extract top n words per topic
    top_words_dict = extract_top_words(lda_model, top_n=40)
    
    # initialize topic names dict
    topic_names = {}
    
    # Assign a topic name for each of the topics 
    for topic_id, top_words in tqdm(top_words_dict.items(), desc="Deciding topic labels..."):
        
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
    
    # Assign topics to the textsx
    df_assigned_topics = create_topics_df(df['text'], lda_model, 
                                          topic_names).drop(columns=["doc_text"])
    
    # Left join the assigned topics to the original dataframe
    df_join = df.merge(df_assigned_topics, on='index')

    # log the completion of the topic assignment
    logging.info("Topics assigned to the texts.")
    
    #############################
    ### 3. Sentiment Analysis ###
    #############################
    
    print("#"*100 + "\n3. Performing Sentiment Analysis...\n" + "#"*100)
    logging.info("Starting sentiment analysis...")

    # create a temporary column for sentiment analysis 
    df_join["processed_text"] = clean_sentiment(df_join['text'], clean_emojis=True)
    
    # create a new column applying the preprocessing 
    df_join = process_with_vader(df_join) # Sentiment analysis with vader
    
    # drop the temp column 
    df_join.drop(columns=["processed_text"], inplace=True)
    
    # log the completion of the sentiment analysis
    logging.info("Sentiment analysis completed.")

    #########################
    ### 4. Trend Analysis ###
    #########################
    
    print("#"*100 + "\n4. Estimating Reddits trend sentiment...\n" + "#"*100)
    logging.info("Starting trend analysis...")
    
    # assign trend sentiment via lexiconds 
    df_join['trend_sentiment'] = df_join['text'].apply(analyze_emotion) # analyzing the trending emotion

    # preview the data 
    print(df_join.head(10))

    # log the completion of the trend sentiment assignment
    logging.info("Trend sentiment assigned.")

    #########################
    ### 5. Stock Analysis ###
    #########################
    
    print("#"*100 + "\n5. Identiying stocks in reddits...\n" + "#"*100)
    logging.info("Starting stock analysis...")
    
    # extract the tickers from all the texts in the df_join and store them in a new column
    df_join['tickers'] = df_join['text'].apply(extract_tickers, 
                                               ticker_df_path="data_raw/russell3000.csv", 
                                               str_format=True)
    
    # preview the data 
    print(df_join.head(10))
    
    # log the completion of the stock identification
    logging.info("Stock identification completed.")
    
    ######################
    ### 6. Save Result ###
    ######################

    print("#"*100 + f"\n6. Saving data to {OUTPUT_PATH}...\n" + "#"*100)
    logging.info(f"Saving the cleaned data to {OUTPUT_PATH}...")

    # Save the cleaned data to a csv file
    df_join.to_csv(f"{OUTPUT_PATH}")    
    
    t1 = time.time() 
    print(f"Took a total of {round(t1-t0, 3)} seconds")




