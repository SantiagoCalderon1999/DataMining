"""This module allows the plotting of clustered instances
"""
import matplotlib.pyplot as plt

def plot_cluster(labels, colors, x, k, centroids = None):
    """Plots the clusterization result
    """
    for i in range(0, k):
        x_boolean = labels == i
        x_filtered = x[x_boolean]
        plt.scatter(x_filtered[:, 0], x_filtered[:, 1],  color=colors[i], marker='+')
        if (centroids):
            plt.scatter(centroids[i][0], centroids[i][1],  color='red', marker = 'o')
    plt.plot()