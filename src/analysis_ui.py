import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data_clean/wsb_clean.csv')

df = load_data()

def parse_tickers(ticker_str):
    if pd.isnull(ticker_str) or not ticker_str:
        return []
    # Assuming tickers are separated by commas and/or spaces, and removing duplicates
    tickers = set(ticker_str.replace(',', ' ').split())
    # Remove any empty strings that may have been added
    tickers.discard('')
    return list(tickers)

df['tickers'] = df['tickers'].apply(parse_tickers)

# Function to get the most common tickers
def get_top_tickers(df, n=10):
    tickers_list = [ticker for sublist in df['tickers'] for ticker in sublist]
    return Counter(tickers_list).most_common(n)

# Creating the visualizations
st.title("WSB Data Analysis")


st.header("Sentiment Score Distribution")
fig, ax = plt.subplots()
sns.histplot(df['sentiment_score'], ax=ax, kde=True)
st.pyplot(fig)

st.header("Posts by Sentiment Label")
fig, ax = plt.subplots()
df['sentiment_label'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

st.header("Top 10 Tickers Mentioned")
top_tickers = get_top_tickers(df, 15)
tickers_df = pd.DataFrame(top_tickers, columns=['Ticker', 'Count'])
fig, ax = plt.subplots()
sns.barplot(data=tickers_df, x='Ticker', y='Count', ax=ax)
st.pyplot(fig)



# Sentiment Score Analysis by Topic Name
topic_sentiment = df.groupby('topic_name')['sentiment_score'].agg(['mean', 'std']).reset_index()
st.header("Sentiment Score by Topic Name")
st.write(topic_sentiment)

# Sentiment Score Analysis by Trend Sentiment
st.header("Sentiment Score for Each Trend Sentiment")
trend_sentiment_groups = df.groupby('trend_sentiment')['sentiment_score']
for name, group in trend_sentiment_groups:
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

# Ticker Analysis - Sentiment Score Distribution
st.header("Search Ticker for Sentiment Score Distribution")
searched_ticker = st.text_input("Enter a ticker to search:", "").upper()

if searched_ticker:
    # Filter posts that mention the searched ticker
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