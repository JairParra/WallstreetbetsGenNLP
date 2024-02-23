"""
utils.py
    This script contains miscellaneous utility functions.

@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# general
import pandas as pd
from tqdm import tqdm


#############################
### 2. LDA Analysis Utils ###
#############################

# Function to format topics and their contribution in each document
def format_topics_sentences(ldamodel=None, corpus=None, texts=None): 

    # Verify that parmeters are not None
    if ldamodel is None or corpus is None or texts is None:
        raise ValueError("The LDA model, corpus, and texts must be provided.")

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