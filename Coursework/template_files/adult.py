"""
    Part 1: Decision Trees with Categorical Attributes
    Functions to perform data mining tasks in a dataset which exhibits information from
    individuals and their income based on 14 attributes
"""

import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Part 1: Decision Trees with Categorical Attributes


# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
    """
            Returns a pandas dataframe given the string where it is stored

    Args:
            data_file (str): direction of the .csv file

    Returns:
            DataFrame: Dataframe read from the provided location
    """
    return pd.read_csv(data_file)


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    """
            Gets number of rows in the pandas dataframe df

    Args:
            df (Dataframe): Dataframe to count the number of rows

    Returns:
            int: Number of rows
    """
    return len(df.index)


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    """
            Gets a list with the attribute names of the pandas dataframe

    Args:
            df (Dataframe): Dataframe to get the list of attributes

    Returns:
            List[str]: List of all the attribute names in the provided dataframe
    """
    return df.columns.tolist()


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    """
            Gets the number of missing attributes in a pandas dataframe

    Args:
            df (Dataframe): Dataframe to get the number of missing values

    Returns:
            int: Number of missing attributes
    """
    df_null_bool = df.isnull()
    return np.count_nonzero(df_null_bool)


# Return a list with the columns names containing at least one missing
# value in the pandas dataframe df.
def columns_with_missing_values(df):
    """
            Returns all the columns with missing attributes

    Args:
            df (Dataframe): Dataframe from which column names with
                            missing attributes will be retrieved

    Returns:
            List[str]: List of attributes with at least one missing value
    """
    series = df.isnull().sum()
    return series[series > 0].keys().to_list()


# Return the percentage of instances corresponding to persons whose education level is
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
    """
            Get percentage of instances corresponding to people whose education
            level is Bachelors or Masters

    Args:
            df (Dataframe): Dataframe to extract attribute percentage

    Returns:
            int: Percentage of people with Bachelors or Masters rounded to the first decimal digit
    """
    education_column_name = "education"
    masters_attribute_name = "Masters"
    bachelors_attribute_name = "Bachelors"

    df_filtered = df[
        (df[education_column_name] == masters_attribute_name)
        | (df[education_column_name] == bachelors_attribute_name)
    ]
    filtered_length = len(df_filtered)
    original_length = len(df)
    return round(filtered_length / original_length * 100, 1)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    """
            Returns a pandas dataframe obtained by removing all instances with at least one
            missing value. Please note that this method returns a deep copy of the original
            dataset. As a result, any modification to the new dataframe will not affect the
            original one.

    Args:
            df (Dataframe): Dataframe to remove instances with missing values

    Returns:
            Dataframe: Dataframe without instances containing missing values
    """
    return df.dropna()


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
    """
            Performs one-hot encoding on an input dataframe

    Args:
            df (Dataframe): Dataframe to perform one-hot encoding

    Returns:
            Dataframe: Dataframe with columns after one-hot encoding,
            excluding the target attribute
    """
    df = data_frame_without_missing_values(df)
    encoded_data = pd.get_dummies(df)
    class_less_50_name = "class_<=50K"
    class_greater_50_name = "class_>50K"
    return encoded_data.loc[
        :,
        (encoded_data.columns != class_less_50_name)
        & (encoded_data.columns != class_greater_50_name),
    ]


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding.
def label_encoding(df):
    """
            Performs label encoding on the target column of the provided dataframe

    Args:
            df (Dataframe): Dataframe to perform label encoding

    Returns:
            Dataframe: Dataframe containing the target variables as numeric values,
            converted via label encoding
    """
    df = data_frame_without_missing_values(df)
    label_encoder = LabelEncoder()
    output_column_name = "class"
    y = label_encoder.fit_transform(df[output_column_name])
    df_encoded = df.copy()
    df_encoded[output_column_name] = y
    return df_encoded[output_column_name]


# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train.
# Return a pandas series with the predicted values.
def dt_predict(X_train, y_train):
    """
            Builds a decision tree and returns Series object with predictions of the
            training dataset

    Args:
            X_train: Features of the training set
            y_train (_type_): Labels of the training instances

    Returns:
            Series: Labels predicted for the given dataset
    """
    decision_tree = tree.DecisionTreeClassifier()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_train)
    label_pred = label_encoder.inverse_transform(y_pred)
    return pd.Series(label_pred)


# Given a pandas series y_pred with the predicted labels
# and a pandas series y_true with the true labels, compute
# the error rate of the classifier that produced y_pred.
def dt_error_rate(y_pred, y_true):
    """
        Gets the error rate of two different pandas series, corresponding to
        the predicted values and the true labels

    Args:
        y_pred (Series): Predicted labels
        y_true (Series): True labels

    Returns:
        float: Error rate
    """
    all_positive = sum(y_pred == y_true)
    instance_number = len(y_true)
    return 1 - all_positive / instance_number
