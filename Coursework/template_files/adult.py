import pandas as pd
from sklearn import tree
# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	return pd.read_csv(data_file)

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return len(df.index)

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return df.columns.tolist()

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().sum().sum() # Look for a better approach, I don't like this sum sum thing.

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	series = df.isnull().sum()
	return series[series>0].keys().to_list()

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	return (len(df[(df['education'] == 'Masters') | (df['education'] == 'Bachelors')]) / len(df) * 100) # Avoid magic strings

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	return df.dropna() # Check if I should use copy

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	encoded_data = pd.get_dummies(df.dropna()) # Should I usde dropna here?
	return encoded_data.loc[:, (encoded_data.columns != 'class_<=50K') & (encoded_data.columns != 'class_>50K')]

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	encoded_data = pd.get_dummies(df.dropna()) # Should I drop the dropna?
	return encoded_data.loc[:, (encoded_data.columns == 'class_<=50K')] # Find a cleaner way to do label encoding

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	decision_tree = tree.DecisionTreeClassifier()
	decision_tree.fit(X_train, y_train)
	return pd.Series(decision_tree.predict(X_train)) # Check if there is a cleaner implementation for this

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	score = 0
	for output in zip(y_pred, y_true):
		score = (score + 1) if output[0] == output[1] else score
	return score / len(y_true) # Ensure that this works and ensure that this is exactly what he means by error rate
