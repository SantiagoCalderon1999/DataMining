"""
    Part 3: Text mining
    Functions to perform text mining tasks
"""

from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

import numpy as np
import requests
import pandas as pd

# Part 3: Text mining.


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
    """
        Returns a pandas dataframe given the string where it is stored,
        encoded with `latin-1`

    Args:
        data_file (str): direction of the .csv file

    Returns:
        DataFrame: Dataframe read from the provided location
    """
    latin_encoding = "latin-1"
    return pd.read_csv(data_file, header=0, encoding=latin_encoding)


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    """
    Extracts unique sentiments from a DataFrame column.

    Args:
        df (DataFrame): A pandas DataFrame containing a column named `Sentiment`.

    Returns:
        List: A list of unique sentiment values extracted from the `Sentiment` column.
    """
    sentiment_column = "Sentiment"
    return list(set(df[sentiment_column]))


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    """
    Returns the second most popular sentiment among the tweets.

    Args:
        df (DataFrame): A pandas DataFrame containing a column named `Sentiment`
                        representing sentiments of tweets.

    Returns:
        str: The second most popular sentiment found in the DataFrame's `Sentiment` column.
    """
    sentiment_column = "Sentiment"
    return df[sentiment_column].value_counts().index[1]


# Return the date (string as it appears in the data) with the greatest
# number of extremely positive tweets.
def date_most_popular_tweets(df):
    """
    Return the date with the greatest number of extremely positive tweets.

    Args:
        df (DataFrame): A pandas DataFrame containing columns `Sentiment` and `TweetAt`.
                        `Sentiment` column represents tweet sentiments, and `TweetAt`
                        column represents tweet dates.

    Returns:
        str: The date (string as it appears in the data) with the greatest number of
             extremely positive tweets.
    """
    sentiment_column = "Sentiment"
    tweet_at_column = "TweetAt"
    extremely_positive_category = "Extremely Positive"
    df_extremely_positive = df[(df[sentiment_column] == extremely_positive_category)]
    return df_extremely_positive[tweet_at_column].value_counts().index[0]


# Modify the dataframe df by converting all tweets to lower case.
def lower_case(df):
    """
    Modifies the DataFrame `df` by converting all tweets in the `OriginalTweet`
    column to lower case.

    Args:
        df (DataFrame): A pandas DataFrame containing a column named `OriginalTweet`
                        representing tweets.

    Returns:
        DataFrame: The modified DataFrame with all tweets in the `OriginalTweet`
                   column converted to lower case.
    """
    original_tweet_column = "OriginalTweet"
    df[original_tweet_column] = df[original_tweet_column].str.lower()
    return df


# Modify the dataframe df by replacing each characters which is not alphabetic
# or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    """
    Modify the DataFrame `df` by replacing each character that is not alphabetic or
    whitespace with a whitespace in the `OriginalTweet` column.

    Args:
        df (DataFrame): A pandas DataFrame containing a column named `OriginalTweet`
        representing tweets.

    Returns:
        DataFrame: The modified DataFrame with non-alphabetic or non-whitespace characters
        replaced by whitespace in the `OriginalTweet` column.
    """
    punctuation_pattern = r"[^A-Za-z\s]"
    space = " "
    original_tweet_column = "OriginalTweet"
    df[original_tweet_column] = df[original_tweet_column].str.replace(
        punctuation_pattern, space, regex=True
    )
    return df


# Modify the dataframe df with tweets after removing characters which are
# not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    """
    Modify the DataFrame `df` by removing characters that are not alphabetic or whitespaces
    from the `OriginalTweet` column.

    Args:
        df (DataFrame): A pandas DataFrame containing a column named `OriginalTweet`
                        representing tweets.

    Returns:
        DataFrame: The modified DataFrame with non-alphabetic or non-whitespace characters removed
                   from the `OriginalTweet` column.
    """
    punctuation_pattern = r"[\s+]"
    space = " "
    original_tweet_column = "OriginalTweet"
    df[original_tweet_column] = df[original_tweet_column].str.replace(
        punctuation_pattern, space, regex=True
    )
    return df


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    """
    Tokenizes each tweet in a DataFrame where each tweet is one string with words
    separated by single whitespaces.

    Args:
        df (DataFrame): DataFrame containing tweets.

    Returns:
        DataFrame: DataFrame with tokenized tweets where each tweet is represented
        as a list of words (strings).
    """
    original_tweet_column = "OriginalTweet"
    df[original_tweet_column] = df[original_tweet_column].str.split()
    return df


# Given dataframe tdf with the tweets tokenized, return the number of words
# in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    """
    Counts the number of words in all tweets including repetitions.

    Args:
        tdf (DataFrame): DataFrame with tokenized tweets.

    Returns:
        int: Total number of words in all tweets.
    """
    original_tweet_column = "OriginalTweet"
    all_elements = [len(sublist) for sublist in tdf[original_tweet_column]]
    return np.array(all_elements).sum()


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    """
    Count the number of distinct words in all tokenized tweets within the given DataFrame.

    Args:
        tdf (DataFrame): DataFrame containing tokenized tweets.

    Returns:
        int: The number of distinct words found across all tweets.
    """
    original_tweet_column = "OriginalTweet"
    all_elements = [item for sublist in tdf[original_tweet_column] for item in sublist]
    return len(set(all_elements))


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
    """
    Return a list with the k distinct words that are most frequently occurring in the
    tokenized tweets DataFrame.

    Args:
        tdf (DataFrame): DataFrame containing tokenized tweets.
        k (int): The number of distinct words to return.

    Returns:
        list: A list containing the k distinct words that are most frequently
        occurring in the tokenized tweets DataFrame.
    """
    original_tweet_column = "OriginalTweet"
    all_elements = [item for sublist in tdf[original_tweet_column] for item in sublist]
    element_counts = Counter(all_elements)
    return [element for element, _ in element_counts.most_common(k)]


# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    """
    Removes stop words and words with <=2 characters from each tweet in the given DataFrame.

    Args:
        tdf (DataFrame): DataFrame with tweets tokenized.

    Returns:
        DataFrame: DataFrame with stop words and short words removed from each tweet.
    """
    original_tweet_column = "OriginalTweet"
    stop_words = (
        requests.get(
            "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt",
            timeout=1000,
        )
        .content.decode("utf-8")
        .splitlines()
    )
    tdf[original_tweet_column] = tdf[original_tweet_column].apply(
        lambda tokens: [
            token for token in tokens if token not in stop_words and len(token) > 2
        ]
    )
    return tdf


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    """
    Reduces each word in every tweet in the given DataFrame to its stem.

    Args:
        tdf (DataFrame): DataFrame with tweets tokenized.

    Returns:
        DataFrame: DataFrame with words in each tweet reduced to their stems.
    """
    original_tweet_column = "OriginalTweet"
    ps = PorterStemmer()
    tdf[original_tweet_column] = tdf[original_tweet_column].apply(
        lambda tokens: [ps.stem(token) for token in tokens]
    )
    return tdf


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier.
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray).
def mnb_predict(df):
    """
    Builds a Multinomial Naive Bayes classifier using the given DataFrame.

    Args:
        df (DataFrame): DataFrame with the original data set.

    Returns:
        numpy.ndarray: Predicted sentiments ('Neutral', 'Positive', etc.) for the training set.
    """
    original_tweet_column = "OriginalTweet"
    sentiment_column = "Sentiment"

    df = remove_non_alphabetic_chars(df)
    df = remove_multiple_consecutive_whitespaces(df)
    count_vectorizer = CountVectorizer(ngram_range=(4, 4))
    x = count_vectorizer.fit_transform(df[original_tweet_column])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[sentiment_column])
    y = np.array(y).ravel()
    clf = MultinomialNB()

    clf.fit(x, y)

    y_pred = clf.predict(x)
    return label_encoder.inverse_transform(y_pred)


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive')
# by a classifier and another 1d array y_true with the true labels,
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
    """
    Calculates the classification accuracy of a classifier.

    Args:
        y_pred (numpy.ndarray): Predicted labels by the classifier.
        y_true (numpy.ndarray): True labels.

    Returns:
        float: Classification accuracy rounded to the 3rd decimal digit.
    """
    all_positive = sum(y_pred == y_true)
    instance_number = len(y_true)
    return round(all_positive / instance_number, 3)
