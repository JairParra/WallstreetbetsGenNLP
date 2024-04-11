"""
analysis_ui.py
    This script contains functions for analyzing sentiment in WSB (WallStreetBets) data.

@author: Cédric Lam 
@author: Hair Parra
"""

################## 
### 1. Imports ###
##################

# Import and configure streamlit 
import streamlit as st

# Set the overall layout to be wider
st.set_page_config(layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# set sns style 
sns.set()

# setup dark plotting
sns.set(style="darkgrid", context="talk")
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

################## 
### 2. Helpers ###
##################

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data_clean/wsb_clean.csv')

def parse_tickers(ticker_str):
    """
    Parses a string of tickers separated by commas or spaces into a list of tickers.

    Parameters:
    - ticker_str (str): The string containing tickers.

    Returns:
    - list: A list of tickers.
    """
    if pd.isnull(ticker_str) or not ticker_str:
        return []
    tickers = set(ticker_str.replace(',', ' ').split())
    tickers.discard('')
    return list(tickers)


def get_top_tickers(df, n=10):
    """
    Finds the most common tickers in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the tickers.
    - n (int): The number of top tickers to return.

    Returns:
    - list: A list of tuples containing the top tickers and their counts.
    """
    tickers_list = [ticker for sublist in df['tickers'] for ticker in sublist]
    return Counter(tickers_list).most_common(n)


################## 
### 3. Dataset ###
##################

# Read local data 
df = load_data()

# Format the tickers from the data 
df['tickers'] = df['tickers'].apply(parse_tickers)

#################### 
### 4. Dashboard ###
####################

# Create the title of the page 
st.markdown("<h1 style='text-align: center;'>WSB Data Analysis</h1>", unsafe_allow_html=True)

# Preview of the first 10 rows of the data
st.header("WSB Analyzed Data")
df_preview = df[["title", "score", "url", "comms_num", "timestamp", "topic_name", 
                 "sentiment_score", "sentiment_label", "trend_sentiment", "tickers"]]
st.write(df_preview)
    
# Create two columns
col1, col2, col3 = st.columns(3)

# Display the first plot in the first column
with col1:
    st.header("Sentiment Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment_score'], ax=ax, kde=True)
    st.pyplot(fig)

# Display the second plot in the second column
with col2:
    st.header("Posts by Sentiment Label")
    fig, ax = plt.subplots()
    sentiment_label_counts = df['sentiment_label'].value_counts()
    sns.barplot(x=sentiment_label_counts.index, y=sentiment_label_counts.values, ax=ax)
    ax.tick_params(axis='x', rotation=0)  # Horizontal labels
    st.pyplot(fig)
    
    
### Top tickers mentioned 
with col3: 
    st.header("Top 10 Tickers Mentioned")
    top_tickers = get_top_tickers(df, 10)
    tickers_df = pd.DataFrame(top_tickers, columns=['Ticker', 'Count'])
    fig, ax = plt.subplots()
    sns.barplot(data=tickers_df, x='Ticker', y='Count', ax=ax)
    plt.xticks(rotation=45)  # Inclined labels
    st.pyplot(fig)
    
    

# Sentiment Score Analysis by Trend Sentiment
st.header("Sentiment Score for Each Trend Sentiment")
trend_sentiment_groups = df.groupby('trend_sentiment')['sentiment_score']

# Create columns for each plot
columns = st.columns(len(trend_sentiment_groups))

# Display each plot in a separate column
for i, (name, group) in enumerate(trend_sentiment_groups):
    with columns[i]:
        # st.header(f"Sentiment Score for Trend: {name}")
        fig, ax = plt.subplots()
        sns.histplot(group, kde=True, ax=ax)
        ax.set_title(f'Trend: {name}')
        mean = group.mean()
        std = group.std()
        ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        ax.axvline(mean + std, color='g', linestyle=':', label=f'Std Dev: {std:.2f}')
        ax.axvline(mean - std, color='g', linestyle=':')
        ax.legend()
        st.pyplot(fig)
        
        
# Create two columns
col1, col2, col3 = st.columns(3)
        
with col2:
    # Ticker Analysis - Sentiment Score Distribution
    st.header("Search Ticker for Sentiment Score Distribution")
    searched_ticker = st.text_input("Enter a ticker to search:", "").upper()
    
    if searched_ticker:
        ticker_posts = df[df['tickers'].apply(lambda x: searched_ticker in x)]
        if not ticker_posts.empty:
            fig, ax = plt.subplots()
            sns.histplot(ticker_posts['sentiment_score'], kde=True, ax=ax)
            ax.set_title(f'Sentiment Score Distribution for {searched_ticker}')
            mean = ticker_posts['sentiment_score'].mean()
            std = ticker_posts['sentiment_score'].std()
            ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
            ax.axvline(mean + std, color='g', linestyle=':', label=f'Std Dev: {std:.2f}')
            ax.axvline(mean - std, color='g', linestyle=':')
            ax.legend()
            st.pyplot(fig)
        else:
            st.write(f"No posts found for ticker {searched_ticker}.")


with col2: 
    
    # Sentiment Score Analysis by Topic Name
    st.header("Sentiment Score by Topic Name")
    topic_sentiment = df.groupby('topic_name')['sentiment_score'].agg(['mean', 'std']).rename(columns={'topic_name': "Topic Name", 
                                                                                                       'mean': "Avg Sentiment"}).reset_index()
    st.table(topic_sentiment.style.set_properties(**{'text-align': 'center'}))