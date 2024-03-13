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
from collections import Counter

# visualization 
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


#############################
### 2. LDA Analysis Utils ###
#############################

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
