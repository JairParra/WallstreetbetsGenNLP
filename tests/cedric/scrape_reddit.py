import praw  # Import the PRAW library for accessing the Reddit API
import pandas as pd  # Import pandas for data manipulation

# Create a Reddit instance
reddit = praw.Reddit(
    client_id="YnmRgUfHOn5foh17UNLsrA",
    client_secret="EcvOf0J1NWVyuF3PTmxGkAAiuqQLkw",
    user_agent="testscript by /u/tailinks",
)

# List to store submission data
data = []

# Fetch the top submissions from the "wallstreetbets" subreddit
for submission in reddit.subreddit("wallstreetbets").top(limit=10000):
    # Create a dictionary to store the submission data
    submission_data = {
        "title": submission.title,
        "score": submission.score,
        "id": submission.id,
        "url": submission.url,
        "num_comments": submission.num_comments,
        "created": submission.created,
    }
    # Append the submission data to the list
    data.append(submission_data)

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Convert the 'created' column to datetime format
df['timestamp'] = pd.to_datetime(df['created'], unit='s')

# Export the DataFrame to a CSV file
df.to_csv('reddit.csv', index=False)
