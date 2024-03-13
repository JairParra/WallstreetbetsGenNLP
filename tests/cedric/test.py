
import time
from ticker_detection import extract_tickers
import pandas as pd

if __name__ == "__main__":
        minimum_market_cap = 1e10
        data = pd.read_csv("data_raw/reddit_wsb.csv")
        
        for index, row in data.head(3).iterrows():
            title = row['title']
            content = row['body']
            if pd.isnull(content):
                content = ""
            if pd.isnull(title):
                title = ""
            tickers = extract_tickers(title + "" +content, minimum_market_cap=minimum_market_cap)
            print(tickers)
            
            time.sleep(1)
            