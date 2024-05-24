import pandas as pd
import datetime as dt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Load the dataset
df = pd.read_csv("../input/amazon-reviews/amazon_reviews.csv")
df.head()

# Dataset shape
df.shape

# Dataset information
df.info()

# Count of different ratings
df["overall"].value_counts()

# Average rating of the product
df["overall"].mean()

# Convert reviewTime to datetime
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

# Get quantiles to determine thresholds
df["day_diff"].quantile([0.25, 0.50, 0.75, 1])

# Function to calculate time-based weighted average
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[(dataframe["day_diff"] <= 281), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 281) & (dataframe["day_diff"] <= 431), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 431) & (dataframe["day_diff"] <= 601), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 601), "overall"].mean() * w4 / 100

# Run the function
time_based_weighted_average(df)

# Compare averages in different time periods
df.loc[(df["day_diff"] <= 281), "overall"].mean()
df.loc[(df["day_diff"] > 281) & (df["day_diff"] <= 431), "overall"].mean()
df.loc[(df["day_diff"] > 431) & (df["day_diff"] <= 601), "overall"].mean()
df.loc[(df["day_diff"] > 601), "overall"].mean()

# It is evident that the rating has increased over time, possibly due to improvements based on feedback.

# Adding helpful_no column since the dataset lacks downvotes
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Up-Down Difference Score
def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = score_up_down_diff(df["helpful_yes"], df["helpful_no"])

# Average Rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# Wilson Lower Bound Score
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Display the top 20 reviews sorted by average rating
df[["score_average_rating", "helpful_yes", "helpful_no", "wilson_lower_bound"]] \
.sort_values("score_average_rating", ascending=False).head(20)
