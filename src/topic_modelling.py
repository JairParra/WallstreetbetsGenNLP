"""
toic_modelling.py
    This script contains functions relating to topic modelling and LDA analysis.

@author: Hair Parra
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

##########################################
### 2. Text Cleaning and Preprocessing ###
##########################################

def clean_text(texts: Union[str, List[str], pd.Series], clean_emojis: bool = False, verbose:bool = False) -> Union[str, List[str]]:
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

#########################
### 3. Model Training ###
#########################

def train_lda_model(clean_texts: List[List[str]], save_model: bool = False) -> gensim.models.ldamodel.LdaModel:
    """
    Train an LDA model on a list of cleaned text documents.

    Parameters:
    - clean_texts: A list of cleaned text documents, with each document represented as a list of tokens.
    - save_model: A boolean indicating whether to save the trained model.

    Returns:
    - The trained LDA model.
    """

    # Create a subset of randomly selected clean texts
    clean_texts_subset = [clean_texts[i] for i in np.random.randint(0, len(clean_texts), len(clean_texts))]

    # Create a dictionary mapping from word IDs to words
    id2word = corpora.Dictionary(clean_texts_subset)

    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count)
    corpus = [id2word.doc2bow(text) for text in clean_texts_subset]

    # Log the perplexity score at the end of each epoch.
    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')

    # Create a dictionary with the parameters used in the LDA model 
    lda_params = {
        'num_topics': 4,                 # The number of requested latent topics to be extracted from the training corpus
        'update_every': 1,               # Number of documents to be iteratively updated
        'chunksize': 100,                 # Number of documents to be used in each training chunk
        'passes': 7,                     # Number of passes through the corpus during training
        'alpha': 'symmetric',            # Hyperparameter affecting sparsity/thickness of the topics
        'iterations': 100,               # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
    }

    # Build LDA model with the corpus and dictionary
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,                              # Corpus to perform LDA on
        id2word=id2word,                            # Mapping from IDs to words
        num_topics=lda_params["num_topics"],        # The number of requested latent topics to be extracted from the training corpus
        random_state=100,                           # Random state for reproducibility
        update_every=lda_params["update_every"],    # Number of documents to be iteratively updated
        chunksize=lda_params["chunksize"],          # Number of documents to be used in each training chunk
        passes=lda_params["chunksize"],             # Number of passes through the corpus during training
        alpha=lda_params['alpha'],                  # Hyperparameter affecting sparsity/thickness of the topics
        iterations=lda_params["iterations"],        # Maximum number of iterations through the corpus when inferring the topic distribution of a corpus
        eval_every=10,                               # Log perplexity is estimated every that many updates
        per_word_topics=True,                       # If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word
        callbacks=[perplexity_logger]               # Log the perplexity score at the end of each epoch
    )

    if save_model:
        # Create a directory for the model if it doesn't exist
        model_dir = 'models/lda_model'
        os.makedirs(model_dir, exist_ok=True)

        # Generate a custom name for the model based on the parameters
        model_name = f'lda_model_{lda_params["num_topics"]}_{lda_params["update_every"]}_{lda_params["chunksize"]}_{lda_params["passes"]}_{lda_params["alpha"]}_{lda_params["iterations"]}.model'

        # Save the model with the custom name
        model_path = os.path.join(model_dir, model_name)
        lda_model.save(model_path)

    return lda_model


def load_lda_model(dir_path: str) -> gensim.models.ldamodel.LdaModel:
    """
    Load the LDA model from the specified directory.

    Parameters:
    - dir_path: The directory path where the LDA model is located.

    Returns:
    - The loaded LDA model.
    """
    # Search for the model file in the directory
    model_files = [f for f in os.listdir(dir_path) if f.endswith('.model')]

    if not model_files:
        raise FileNotFoundError("No model file found in the specified directory.")

    # Load the model
    model_path = os.path.join(dir_path, model_files[0])
    lda_model = gensim.models.LdaModel.load(model_path)

    return lda_model



####################
### 4. LDA Utils ###
####################


def assign_topic(raw_text: str, lda_model: LdaModel, return_all: bool = False, verbose: bool = False) -> Union[Tuple[int, float], List[Tuple[int, float]]]:
    """
    Assigns a topic to a given raw text using an LDA model.

    Args:
        raw_text (str): The raw text to assign a topic to.
        lda_model (LdaModel): The trained LDA model.
        return_all (bool, optional): Whether to return all topics and their probabilities. Defaults to False.
        verbose (bool, optional): Whether to print the original and cleaned text. Defaults to False.

    Returns:
        Union[Tuple[int, float], List[Tuple[int, float]]]: The most likely topic and its probability, or a list of all topics and their probabilities.
    """
    # Preprocess the raw text
    cleaned_text = clean_text([raw_text], clean_emojis=True, verbose=False)[0]

    # Show text before and after cleaning
    if verbose:
        print(f"Original text: {raw_text}")
        print(f"Cleaned text: {' '.join(cleaned_text)}")

    # Convert the cleaned text into bag-of-words format
    bow = lda_model.id2word.doc2bow(cleaned_text)
    
    # Get the topic distribution for the text
    topic_distribution = lda_model.get_document_topics(bow)
    
    # Sort the topic distribution by probabilities in descending order
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
    return sorted_topics if return_all else sorted_topics[0]



def create_topics_df(texts: List[str], lda_model: LdaModel) -> pd.DataFrame:
    """
    Assign topics to a list of texts and create a dataframe with the assigned topics.

    Parameters:
        texts: A list of text documents.
        lda_model (LdaModel): The trained LDA model.

    Returns:
        A pandas DataFrame with the assigned topics for each text document.
    """

    # Assign topics to the texts
    assigned_topics = [assign_topic(doc, lda_model, return_all=True) for doc in texts]

    # Create a list of records using list comprehensions
    records = [{
        'doc_text': doc_text,
        'most_likely_topic': topics_i[0][0],
        'probability': topics_i[0][1],
        'topics': [tup[0] for tup in topics_i]
    } for doc_text, topics_i in zip(df['text'].iloc[:10], assigned_topics)]

    # Create a dataframe from the records
    df_assigned_topics = pd.DataFrame(records)
    
    return df_assigned_topics



# Function to format topics and their contribution in each document
def format_topics_sentences(ldamodel=None, corpus=None, texts=None): 

    # Verify that parmeters are not None
    if ldamodel is None or corpus is None or texts is None:
        raise ValueError("The LDA model, corpus, and texts must be provided.")
    
    # verify that corpus and texts have the same length
    if len(corpus) != len(texts):
        raise ValueError("The corpus and texts must have the same length.")

    # Initialize a list to store each document's dominant topic and its properties
    records = []

    # Iterate over each document in the corpus
    for i, row_list in tqdm(enumerate(ldamodel[corpus]), desc="iterating through corpus...", total=len(corpus)):

        # Check if the model has per word topics or not to choose the correct element
        row = row_list[0] if ldamodel.per_word_topics else row_list

        # Sort each document's topics by the percentage contribution in descending order
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Extract the dominant topic and its percentage contribution for each document
        for j, (topic_num, prop_topic) in enumerate(row):

            # Only the top topic (dominant topic) is considered
            if j == 0:

                # Get the topic words and weights
                wp = ldamodel.show_topic(topic_num)

                # Join the topic words
                topic_keywords = ", ".join([word for word, prop in wp])

                # Create the records
                record = (int(topic_num), round(prop_topic, 4), topic_keywords)

                # Append the dominant topic and its properties to the list
                records.append(record)

                # Exit the loop after the dominant topic is found
                break

    # Create the DataFrame from the accumulated rows
    sent_topics_df = pd.DataFrame(records, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    # Add the original text of the documents to the DataFrame
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # Reset the index of the DataFrame for aesthetics and readability
    sent_topics_df = sent_topics_df.reset_index()

    # Rename the columns of the DataFrame for clarity
    sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    return sent_topics_df


########################
### 5. Visualization ###
########################

def plot_topic_keywords(lda_model, clean_texts):

    # Extract topics and flatten data
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in clean_texts for w in w_list]
    counter = Counter(data_flat)

    # Initialize empty list to store data
    out = []

    # Iterate over topics and their words to retrieve the weights and
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    # Create DataFrame from collected data
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)

    # Define colors for each subplot
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    # Iterate over subplots
    for i, ax in enumerate(axes.flatten()):
        # Plot bar chart for word count
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')

        # Create twinx axis for importance
        ax_twin = ax.twinx()

        # Plot bar chart for word importance
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')

        # Set y-axis labels
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030)
        ax.set_ylim(0, 3500)

        # Set title for subplot
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)

        # Hide y-axis ticks
        ax.tick_params(axis='y', left=False)

        # Rotate x-axis labels
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')

        # Add legends
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    # Adjust layout
    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)

    # Show plot
    plt.show()
