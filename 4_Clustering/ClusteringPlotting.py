import matplotlib.pyplot as plt

def plot_cluster(labels, colors, X, k, centroids = []):
    for i in range(0, k):
        X_boolean = labels == i
        X_filtered = X[X_boolean]
        plt.scatter(X_filtered[:, 0], X_filtered[:, 1],  color=colors[i], marker='+')
        if (centroids):
            plt.scatter(centroids[i][0], centroids[i][1],  color='red', marker = 'o')
    plt.plot()