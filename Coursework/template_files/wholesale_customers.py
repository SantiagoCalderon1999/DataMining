"""
    Part 2: Cluster Analysis
    Functions to perform clustering in a dataset containing information 
    about wholesale costumers
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics

# Part 2: Cluster Analysis


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    """
    Return a pandas DataFrame from the specified file location.

    Args:
        data_file (str): Path to the .csv file.

    Returns:
        DataFrame: DataFrame read from the provided location.
    """

    channel_column = "Channel"
    region_column = "Region"
    return pd.read_csv(
        data_file, usecols=lambda x: (x != channel_column) & (x != region_column)
    )


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed
# with the attribute name.
def summary_statistics(df):
    """
    Return a pandas DataFrame with summary statistics of the provided DataFrame.
    Note: This function employs a vectorized approach to enhance the algorithm's runtime. Additionally,
        it inserts the results into a dictionary first to further improve performance.

    Args:
        df (DataFrame): DataFrame to extract statistics.

    Returns:
        DataFrame: DataFrame containing columns for `mean`, `std`, `min`, and `max`.
                Each row represents an attribute in the provided dataset.
    """

    summary = {}
    mean_column = "mean"
    std_column = "std"
    min_column = "min"
    max_column = "max"
    summary[mean_column] = round(df.mean()).astype(np.int64)
    summary[std_column] = round(df.std()).astype(np.int64)
    summary[min_column] = df.min()
    summary[max_column] = df.max()
    return pd.DataFrame(summary)


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    """
    Standardize a given dataset by subtracting the mean and dividing by the standard deviation of each attribute.
    Note: This function employs a vectorized approach to enhance the algorithm's runtime.

    Args:
        df (DataFrame): DataFrame to standardize.

    Returns:
        DataFrame: Standardized DataFrame.
    """
    standardized_data = df.copy()
    mean = standardized_data.mean()
    std = standardized_data.std()
    standardized_data = (standardized_data - mean) / std
    return standardized_data


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
    """
    Perform K-means clustering algorithm given a DataFrame and a number of clusters.
    Note: The initial centroids are initialized randomly as per the original K-means algorithm.

    Args:
        df (DataFrame): DataFrame containing the instances.
        k (int): Number of clusters.

    Returns:
        Series: Series containing the labels produced by the clustering of the given DataFrame.
    """
    kmeans_model = cluster.KMeans(n_clusters=k, init="random")
    kmeans_model.fit(df)
    return pd.Series(kmeans_model.labels_)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    """
    Perform K-means clustering algorithm given a DataFrame and a number of clusters.
    Note: The initial centroids are initialized using the kmeans++ algorithm.

    Args:
        df (DataFrame): DataFrame containing the instances.
        k (int): Number of clusters.

    Returns:
        Series: Series containing the labels produced by the clustering of the given DataFrame.
    """
    kmeans_plus_model = cluster.KMeans(n_clusters=k, init="k-means++")
    kmeans_plus_model.fit(df)
    return pd.Series(kmeans_plus_model.labels_)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    """
    Perform agglomerative clustering algorithm given a DataFrame and a number of clusters.

    Args:
        df (DataFrame): DataFrame containing the instances.
        k (int): Number of clusters.

    Returns:
        Series: Series containing the labels produced by the clustering of the given DataFrame.
    """
    agg = cluster.AgglomerativeClustering(
        n_clusters=k, linkage="single", affinity="euclidean"
    )
    agg.fit(df)
    return pd.Series(agg.labels_)


# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X, y):
    """
    Compute the Silhouette score given a dataset with attributes and its
    corresponding cluster assignments.

    Args:
        X (DataFrame): DataFrame containing the instances.
        y (Series): Series containing cluster assignments.

    Returns:
        float: Silhouette score of the given dataset and labels.
    """
    return metrics.silhouette_score(X, y, metric="euclidean")


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the:
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative',
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    """
    Evaluates different clustering algorithms on a DataFrame.

    This function performs clustering using two algorithms, namely `kmeans` and
    `agglomerative`, and evaluates their performance across multiple configurations.
    Specifically, it performs:

    - `Kmeans` clustering 10 times for each of the specified numbers of clusters
    (3, 5, and 10) on both the original and standardized dataset.

    - `Agglomerative` clustering once for each of the specified numbers of clusters
    (3, 5, and 10) on both the original and standardized dataset.

    Args:
            df (Dataframe): Dataframe to perform clustering

    Returns:
            Dataframe: Dataframe including the following columns:
            `Algorithm` name: either 'Kmeans' or 'Agglomerative',
            `data` type: either 'Original' or 'Standardized',
            `k`: the number of clusters produced,
            `Silhouette Score`: for evaluating the resulting set of clusters.
    """
    cluster_numbers = [3, 5, 10]
    data_variants = [("Original", df.copy()), ("Standardized", standardize(df.copy()))]
    clustering_algorithms = [
        ("Kmeans", kmeans, 10),  # 10 iterations for kmeans
        ("Agglomerative", agglomerative, 1),  # 1 iteration for agglomerative clustering
    ]
    clustering_results = []
    for data_label, dataframe in data_variants:
        for (
            algorithm_name,
            algorithm_function,
            num_iterations,
        ) in clustering_algorithms:
            for cluster_num in cluster_numbers:
                for _ in range(num_iterations):
                    cluster_output = algorithm_function(dataframe, cluster_num)
                    silhouette_score = clustering_score(dataframe, cluster_output)
                    clustering_results.append(
                        [algorithm_name, data_label, cluster_num, silhouette_score]
                    )

    return pd.DataFrame(
        clustering_results, columns=["Algorithm", "data", "k", "Silhouette Score"]
    )


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    """
    Computes the best clustering score among all the implemented algorithms.

    Args:
        rdf (DataFrame): Performance evaluation dataframe including
                        the `Silhouette score` column.

    Returns:
        Float: Best Silhouette score.
    """
    silhouette_score_column = "Silhouette Score"
    return max(rdf[silhouette_score_column])


# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    """
    Performs K-means clustering with k=3 and stores plots with the naming
    convention `{attribute_1}--{attribute_2}.pdf`.

    Parameters:
        df (DataFrame): DataFrame to compute K-means clustering.

    Returns:
        None
    """

    k_clusters = 3
    kmeans_model = cluster.KMeans(n_clusters=k_clusters)

    df = standardize(df)
    kmeans_model.fit(df)

    for i, attr_1 in enumerate(df.columns):
        for j in range(i + 1, len(df.columns)):
            attr_2 = df.columns[j]
            data = df.loc[:, (df.columns == attr_1) | (df.columns == attr_2)]
            cluster_nums = np.array(range(k_clusters))
            cluster_names = np.char.add("Cluster ", cluster_nums.astype(str))
            for cluster_index in cluster_nums:
                data_mask = kmeans_model.labels_ == cluster_index
                data_filtered = data[data_mask]
                plt.scatter(data_filtered[attr_1], data_filtered[attr_2], marker="+")
            plt.xlabel(attr_1)
            plt.ylabel(attr_2)
            plt.title(f"{attr_1} vs {attr_2}")
            plt.legend(cluster_names)
            plt.savefig(f"{attr_1}--{attr_2}.pdf", dpi=500)
            plt.show()
