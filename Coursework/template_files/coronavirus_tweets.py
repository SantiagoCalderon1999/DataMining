import pandas as pd
# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	return pd.read_csv(data_file, header=0, encoding='latin-1')

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return list(set(df['Sentiment']))

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	possible_sentiments = set(df['Sentiment'])
	print(f'Possible sentiments: {set(df["Sentiment"])}')

	dict_sent= {}
	max = 0
	sec_max=0
	sec_max_label = ''
	max_label = ''
	for sentiment in possible_sentiments:
		length = len(df['Sentiment'].loc[(df['Sentiment']==sentiment)])
		dict_sent[sentiment] = length
		if (length>max):
			sec_max=max
			sec_max_label=max_label
			max_label = sentiment
			max=length
	return sec_max_label # ENSURE THAT THE SECOND MAXIMUM IS THE RETURNED VALUE!!!

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	possible_dates = set(df['TweetAt'].loc[(df['Sentiment']=='Extremely Positive')])
	max = 0
	max_date = ''
	max_date_dict = {}
	for date in possible_dates:
		length = len(df['Sentiment'].loc[(df['Sentiment'] =='Extremely Positive') & (df['TweetAt'] == date)])
		max_date_dict[date] = length
		if (length>max):
			max_date=date
			max=length
	return max_date

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: (x.lower()))
	return df

import re
def remove_punctuation_space(text): # Remove this one?
    # Define the regex pattern to match punctuation
    punctuation_pattern = r'[^\w\s]'
    # Replace punctuation with an empty string
    clean_text = re.sub(punctuation_pattern, ' ', text)
    return clean_text

import re
def remove_multiple_white_spaces(text): # Remove this one?
    # Define the regex pattern to match punctuation
    punctuation_pattern = r'[\s*]'
    # Replace punctuation with an empty string
    clean_text = re.sub(punctuation_pattern, ' ', text) # Fix!, it is still returning the spaces
    return clean_text


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: remove_punctuation_space(x))
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: remove_multiple_white_spaces(x))
	return df

import nltk

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	nltk.download('punkt')
	df_new = pd.DataFrame()
	df_new['Tokens'] = df['OriginalTweet'].apply(lambda x: nltk.word_tokenize(x))
	return df_new

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	return tdf['Tokens'].apply(len).sum()

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	all_elements = [item for sublist in tdf['Tokens'] for item in sublist]
	return len(set(all_elements))
from collections import Counter

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	all_elements = [item for sublist in tdf['Tokens'] for item in sublist]
	element_counts = Counter(all_elements)
	return [element for element, _ in element_counts.most_common(k)]

import requests
# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stopwords = requests.get("https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt").content.decode('utf 8').split ("\n")
	tdf['Tokens'] = tdf['Tokens'].apply(lambda tokens: [token for token in tokens if token not in stopwords and len(token) >=2])
	return tdf

from nltk.stem import PorterStemmer

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	ps = PorterStemmer()
	tdf['Tokens'] = tdf['Tokens'].apply(lambda tokens: [ps.stem(token) for token in tokens])
	return tdf

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	import numpy as np
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import CountVectorizer
	import pandas as pd
	df1 = remove_non_alphabetic_chars(df)
	df1 = remove_multiple_consecutive_whitespaces(df)
	#df1 = coronavirus_tweets.tokenize(df1)
	#df1 = coronavirus_tweets.remove_stop_words(df1)

	coun_vect = CountVectorizer(max_features=50)
	count_matrix = coun_vect.fit_transform(df['OriginalTweet'])
	count_array = count_matrix.toarray()
	df2 = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names_out())

	from sklearn.preprocessing import LabelEncoder

	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(df['Sentiment'])
	y = np.array(y).ravel()
	x = df2.values
	X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 0)
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(x)
	return label_encoder.inverse_transform(y_pred)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
    all_positive = sum(y_pred == y_true)
    instance_number = len(y_true)
    return all_positive / instance_number






