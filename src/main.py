"""
main.py
    This script constitues the entrypoint to running one iteration of our project. 

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# general 
import pprint 
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
from src.topic_modelling import train_lda_model 
from src.topic_modelling import load_lda_model
from src.topic_modelling import extract_top_words
from src.topic_modelling import create_topics_df

# supress all warnings from praw 
warnings.filterwarnings('ignore', module='praw')

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint

# Change number of display columns
pd.options.display.max_columns = None

# default configurations 
LOAD_SAMPLE_DATA = False
RETRAIN = False
verbose = 1

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
    
    if verbose >= 1:
        # Print the top n words for each topic by contactenating them with comas 
        for topic_id, top_words in top_words_dict.items():
            print(f"Topic {topic_id}: {', '.join(top_words)}")
        
    # Logic to assign the actual topics 
    # TODO: Implement a function to assign the actual topic names based on top_words_dict
    topic_names = {
        0: "Topic1",
        1: "Topic2",
        2: "Topic3",
        3: "Topic4"
    }
    
    # Assign topics to the texts
    df_assigned_topics = create_topics_df(df['text'], lda_model).drop(columns=["doc_text", "topics"])
    print(df_assigned_topics)
    
    # Left join the assigned topics to the original dataframe
    df_join = df.merge(df_assigned_topics, left_index=True, right_index=True)
    
    
    
        
    







