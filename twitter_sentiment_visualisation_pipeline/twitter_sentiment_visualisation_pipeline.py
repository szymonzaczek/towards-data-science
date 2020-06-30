import twitter
import os
import datetime
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

api = twitter.Api(
    consumer_key=os.environ.get("consumer_key"),
    consumer_secret=os.environ.get("consumer_secret"),
    access_token_key=os.environ.get("access_token_key"),
    access_token_secret=os.environ.get("access_token_secret"),
    tweet_mode="extended",
)

keywords = ["crude", "oil"]

date_for_query = datetime.date.today()

while True:
    date_formatted = date_for_query.strftime("%Y-%m-%d")
    query = (
        "q="
        + "%20".join(keywords)
        + "%20since%3A"
        + date_formatted
        + "%20"
        + "-filter%3Alinks%20filter%3Areplies&count=100&lang=en"
    )
    search_results = api.GetSearch(raw_query=query)
    if len(search_results) > 30:
        break
    elif date_for_query < datetime.date.today() - datetime.timedelta(days=7):
        raise ValueError(
            f"In the last week there has not been enough tweets to produce valuable sentiment analysis for following keywords: {', '.join(keywords)}. Please, try again with different keywords or a lesser amount of keywords, since all of the specified keywords must appear in the tweet for it to be considered for analysis."
        )
    else:
        date_for_query = date_for_query - datetime.timedelta(days=1)

tweets_text = []
tweets_authors = []

for tweet in search_results:
    # CAVEAT: consider only a single tweet from one author - this will ensure that a total sentiment is an opinion of multiple tweeter's users, not just a single one
    if tweet.user not in tweets_authors:
        tweets_authors.append(tweet.user)
    else:
        continue
    tweet_text = tweet.full_text
    # CAVEAT: getting rid of hashtags and @ - no reason to subject it to sentiment analysis
    if "@" in tweet_text:
        tweet_text = re.sub("(@)\w+", "", tweet_text)
    if "#" in tweet_text:
        tweet_text = re.sub("(#)\w+", "", tweet_text)
    if len(tweet_text) > 20:
        tweets_text.append(tweet_text.lstrip())


def vader_sentiment_score(tweet_text: str) -> float:
    return round(
        SentimentIntensityAnalyzer().polarity_scores(tweet_text)["compound"], 3
    )


sentiment_score_vader = []
for tweet in tweets_text:
    sentiment_score_vader.append(vader_sentiment_score(tweet))

sentiment_list = []
for score in sentiment_score_vader:
    if score < -0.05:
        sentiment_list.append("negative")
    elif -0.05 <= score <= 0.05:
        sentiment_list.append("neutral")
    elif score > 0.05:
        sentiment_list.append("positive")

x_axis = ["negative", "neutral", "positive"]
y_axis = [
    sentiment_list.count("negative"),
    sentiment_list.count("neutral"),
    sentiment_list.count("positive"),
]

colors = ["firebrick", "dodgerblue", "limegreen"]
sns.set_palette(sns.color_palette(colors))
sns.set_style("darkgrid")
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = plt.axes()
ax.set_ylabel("Amount", weight="bold", fontsize="large")
sns.barplot(x=x_axis, y=y_axis)
if date_for_query != datetime.date.today():
    fig.suptitle(
        f"Sentiment of Tweets containing keywords: {', '.join(keywords).replace('[', '').replace(']', '')}\ndated for period: {date_for_query.strftime('%d.%m')} - {datetime.date.today().strftime('%d.%m')}",
        weight="bold",
    )
else:
    fig.suptitle(
        f"Sentiment of Tweets containing keywords: {', '.join(keywords).replace('[', '').replace(']', '')}\npublished today ({date_for_query.strftime('%d.%m')})",
        weight="bold",
    )
ylim = max(*y_axis) + 0.1 * max(*y_axis)
for p in ax.patches:
    b = p.get_bbox()
    val = f"{int(b.y1 + b.y0)}"
    y_offset = 0.3
    if int(val) < 10:
        x_offset = -0.04
    else:
        x_offset = -0.08
    ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), weight="bold")
    ax.set_ylim(plt.axes().get_ylim()[0], ylim + 0.5)

fig.savefig(
    f"tweets_sentiment_{'_'.join(keywords).replace('[', '').replace(']', '')}_{datetime.date.today().strftime('%d_%m')}.png",
    bbox_inches="tight",
    pad_inches=0.1,
)
