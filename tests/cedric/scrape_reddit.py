import praw
import pandas as pd

# Create a Reddit instance
reddit = praw.Reddit(
    client_id="YnmRgUfHOn5foh17UNLsrA",
    client_secret="EcvOf0J1NWVyuF3PTmxGkAAiuqQLkw",
    user_agent="testscript by /u/tailinks",
)

data = []
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
df['timestamp'] = pd.to_datetime(df['created'], unit='s')
# Print the DataFrame
df.to_csv('reddit.csv',index=False)
    