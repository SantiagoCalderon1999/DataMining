"""
    Part 2: Cluster Analysis
    Functions to perform clustering in a dataset containing information 
    about wholesale costumers
"""

import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import numpy as np

# Part 2: Cluster Analysis


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    data = pd.read_csv(data_file)
    data = data.loc[:, (data.columns != "Channel") & (data.columns != "Region")]
    return data


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    data_statistics = []
    for column in df.columns:  # Is there a better way than just using the for?
        data_statistics.append(
            [
                round(df[column].mean()),
                round(df[column].std()),
                df[column].min(),
                df[column].max(),
            ]
        )  # Ensure round works fine
    data = pd.DataFrame(
        data_statistics, columns=["mean", "std", "min", "max"]
    ).set_index(df.columns)
    return data


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    standardized_data = df.copy()
    for column in df.columns:
        standardized_data[column] = (
            standardized_data[column] - df[column].mean()
        ) / df[column].std()
    return standardized_data


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
    km = cluster.KMeans(n_clusters=k)
    km.fit(df)
    return pd.Series(km.labels_)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    kmpp = cluster.KMeans(n_clusters=k, init="k-means++")
    kmpp.fit(df)
    return pd.Series(kmpp.labels_)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    agg = cluster.AgglomerativeClustering(
        n_clusters=k, linkage="average"
    )  # Add euclidean distance
    agg.fit(df)
    return pd.Series(agg.labels_)


# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X, y):
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
    ks = [3, 5, 10]
    number_iterations = 10
    results = []
    data_types = ["Original", "Standardized"]
    for data_type in data_types:
        if data_types == "Original":
            df_used = df.copy()
        else:
            df_used = standardize(df.copy())

        for k in ks:
            SC_cummulative = []
            for i in range(0, number_iterations):
                km = cluster.KMeans(n_clusters=k)

                km.fit(df_used)
                SC = metrics.silhouette_score(df_used, km.labels_, metric="euclidean")
                results.append(["Kmeans", data_type, k, SC])
        for k in ks:
            # km = cluster.AgglomerativeClustering(n_clusters = k, linkage='average', metric='euclidean')
            km = cluster.AgglomerativeClustering(
                n_clusters=k, linkage="single"
            )  # See if Euclidean metric can now be used

            km.fit(df_used)

            SC = metrics.silhouette_score(df_used, km.labels_, metric="euclidean")

            results.append(["Agglomerative", data_type, k, SC])
    return pd.DataFrame(results, columns=["Algorithm", "data", "k", "Silhouette Score"])


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    return max(rdf["Silhouette Score"])


# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    k = 3
    km = cluster.KMeans(n_clusters=k)
    df = standardize(df)
    km.fit(df)
    import numpy as np

    colors = []
    for i in range(0, k):
        colors.append(np.random.rand(3, 1).flatten())

    for i in range(0, len(df.columns)):
        label_1 = df.columns[i]
        for j in range(i + 1, len(df.columns)):
            label_2 = df.columns[j]
            _plot_cluster(
                km.labels_,
                colors,
                df.loc[:, (df.columns == label_1) | (df.columns == label_2)],
                k,
                label_1,
                label_2,
            )


import matplotlib.pyplot as plt


def _plot_cluster(labels, colors, x, k, label_1, label_2):
    """Plots the clusterization result"""
    for i in range(0, k):
        x_boolean = labels == i
        x_filtered = x[x_boolean].to_numpy()
        plt.scatter(x_filtered[:, 0], x_filtered[:, 1], color=colors[i], marker="+")
    plt.title(f"{label_1} vs {label_2}")
    plt.plot()
    plt.show()
