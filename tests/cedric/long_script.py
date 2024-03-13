# %% [markdown]
# # Wallstreetbets data (Kaggle) EDA

# %% [markdown]
# ### Parent Directory Configurations 
# 
# **Notebook note:** Please make sure that the `PATH` of this notebook corresponds to the base-directory path of this repository. 
# This will ensure that all executions, data reading will have the reference as if this notebook were in the base directory; 
# otherwise, you might need to change the PATH and do some trickery (which is a pain for Jupyter notebooks) 

# %%
# verify working directory of the notebook 
import os 
print(os.getcwd())

# %% [markdown]
# ## Imports 

# %%
# General imports
import pprint 
import zipfile 
import logging
from typing import Union, List 
from tqdm.notebook import tqdm 

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

import en_core_web_sm

nlp = en_core_web_sm.load()
# Text Processing 
import re 
import spacy
import gensim
from gensim import corpora
from emoji import demojize
from spacy.tokens import Doc
from nltk.corpus import stopwords
from gensim.models.callbacks import PerplexityMetric
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

# Dedicated NLP Visualizations 
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud

# Custom scripts 
from utils import format_topics_sentences
from utils import plot_topic_keywords

# Configurations 
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Exclude common negation words from the stop words list
# negation_words = {'no', 'not', 'nor', 'neither', 'never', "n't", 'none', 'through'}
except_words = {'through'}
stop_words = stop_words - except_words

# nlp.max_length = 1500000  # Adjust based on your text size

# Set pprint options with indent 4
pprint = pprint.PrettyPrinter(indent=4).pprint

# %% [markdown]
# # Data Loading 

# %% [markdown]
# ### Extracting the data 
# 
# Here, we want to extract and preview the data 

# %%
# Read the CSV using Pandas
csv_file_path = '/Users/ced/Documents/WallstreetbetsGenNLP/data_raw/reddit_wsb.csv'
df = pd.read_csv(csv_file_path)

# Fill all the NaN values in the body column with an empty string
df['body'] = df['body'].fillna('')

# Combine the title and bodyy into a single column text, separated by two newlines
df['text'] = df['title'] + '\n\n' + df['body']

# drop the body column 
df = df.drop(columns=['body'])

# %%
print(df.shape)
print(df.columns)

# %%
texts = df['title'].iloc[0:10, ].tolist()
texts

# %% [markdown]
# ### Text Cleaning (for Clustering)

# %%
def clean_text(texts:Union[str, List[str], pd.Series], clean_emojis:bool=False) -> Union[str, List[str]]:

    # Create a list to store the cleaned texts
    cleaned_texts = []

    # Go through every text in the iput list of texts
    for doc in tqdm(nlp.pipe(texts, batch_size=50), 
                             total=len(texts), desc="Cleaning Texts"): 
        
        # print("Original text: ", doc)
        
        # Demojize the token.lemma for each token if it exists, else the token.text 
        tokens = [demojize(token.lemma_ if token.lemma_ != '-PRON-' else token.text).lower() for token in doc]

        # Convert emojis of form :emojiname: to words in format emojiEmojiName
        tokens = [re.sub(r':', '_', token) if token.startswith(':') and token.endswith(':') else token for token in tokens]

        # Remove emojis if prompted 
        if clean_emojis:
            tokens = [re.sub(r'_.*_', '', token) for token in tokens]

        # Remove non-alphabetic characters except for _ 
        tokens = [re.sub(r'[^a-z_]', '', token) for token in tokens]

        # Remove stopwordsm empty tokens and tokens with length less than 2
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

        # # Join tokens that start with "no" or "not" to the next token, but preserve the original token too
        # tokens = [tokens[i] + '_' + tokens[i+1] if tokens[i] in negation_words else tokens[i] for i in range(len(tokens)-1)]
        
        # Append token to the cleaned_texts list
        cleaned_texts.append(tokens)

    # Form bigrams and trigrams models
    bigram = gensim.models.Phrases(cleaned_texts, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)  # Create bigrams with a high threshold for fewer phrases
    trigram = gensim.models.Phrases(bigram[cleaned_texts], threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)  # Create trigrams based on the bigrams
    bigram_mod = gensim.models.phrases.Phraser(bigram)  # Convert bigram model into a more efficient Phraser object
    trigram_mod = gensim.models.phrases.Phraser(trigram)  # Convert trigram model into a Phraser object for efficiency

    # Form bigrams and trigrams
    cleaned_texts = [bigram_mod[doc] for doc in tqdm(cleaned_texts, desc="creating bigrams...")]
    cleaned_texts = [trigram_mod[bigram_mod[doc]] for doc in tqdm(cleaned_texts, desc="creating trigrams...")]

    return cleaned_texts


# %%
# Extract all the titles from the dataframe
texts = df['title'].tolist()

# Clean the corpus
clean_texts = clean_text(texts, clean_emojis=True)

# %%
# Display the cleaned corpus
for i, document in enumerate(clean_texts): 
    if i < 10: 
        print("original doc: \t", texts[i])
        print("clean doc: \t", document)
    else: 
        break

# %% [markdown]
# # LDA Modelisation 
# 
# 

# %%
###############################################
### Step 1: Preparation and hyperparameters ###
###############################################

# Create a subset of randomly selected clean texts
clean_texts_subset = [clean_texts[i] for i in np.random.randint(0, len(clean_texts), max(50000, len(clean_texts)))]

# Create a dictionary mapping from word IDs to words
id2word = corpora.Dictionary(clean_texts_subset)

# Convert document into the bag-of-words (BoW) format = list of (token_id, token_count)
corpus = [id2word.doc2bow(text) for text in clean_texts_subset]

# Log the perplexity score at the end of each epoch.
perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')

# Define the ranges for the parameters
num_topics_options = [2,3,4]  # Adjust if you want to explore other numbers of topics
update_every_options = [2, 3, 4]
chunksize_options = [25, 50, 100]
alpha_options = ['symmetric', 'asymmetric']

# Directory for saving models
model_dir = 'models/lda_models'
os.makedirs(model_dir, exist_ok=True)

# Iterate over the configurations
for num_topics in num_topics_options:
    for update_every in update_every_options:
        for chunksize in chunksize_options:
            for alpha in alpha_options:
                # Define LDA model parameters for the current configuration
                lda_params = {
                    'num_topics': num_topics,
                    'update_every': update_every,
                    'chunksize': chunksize,
                    'passes': 5,  # Keeping this constant as per your script
                    'alpha': alpha,
                    'iterations': 100,  # Keeping this constant as well
                }
                
                # Build LDA model
                lda_model = gensim.models.ldamodel.LdaModel(
                    corpus=corpus,
                    id2word=id2word,
                    num_topics=lda_params["num_topics"],
                    random_state=100,
                    update_every=lda_params["update_every"],
                    chunksize=lda_params["chunksize"],
                    passes=lda_params["passes"],
                    alpha=lda_params['alpha'],
                    iterations=lda_params["iterations"],
                    eval_every=1,
                    per_word_topics=True,
                    callbacks=[perplexity_logger]
                )

                # Generate a custom name for the model based on the parameters
                model_name = f'lda_model_{num_topics}_{update_every}_{chunksize}_{lda_params["passes"]}_{alpha}_{lda_params["iterations"]}.model'
                
                # Save the model with the custom name
                model_path = os.path.join(model_dir, model_name)
                lda_model.save(model_path)

                print(f"Model saved: {model_name}")