# Data Mining Techniques ![Python](https://img.shields.io/badge/python->=3.8-blue.svg)

Welcome to the Data Mining Techniques repository! Here, you'll find a collection of notebooks showcasing various data mining techniques implemented in Python.

## Getting Started

1. Make sure you have Python 3.8 or higher installed on your local machine.
2. Install the required dependencies.

```shell
pip install -r requirements.txt
```

3. Explore the notebooks at your liking!

## Linear Models

These type of models are suitable for either performing regression tasks on datasets that can be modeled as a straight line or classification on linearly separable datasets.

### Linear Classification
In this notebook, we delve into linear classification techniques. We generate two clusters of data randomly, each drawn over a straight line. Then, we separate these clusters using the Perceptron algorithm.

[notebook](linear_models/linear_classification.ipynb)

### Linear Regression
Explore linear regression using Gradient Descent! We start by manually implementing Gradient Descent on the London borough profiles dataset to predict the maximum.

![Linear Regression](images/LinearRegression.png)

[notebook](linear_models/linear_regression.ipynb)

## Nominal Data Classification

This section delves into various algorithms tailored for classifying nominal data, providing insights into their applications and effectiveness.

### OneR (1R) Algorithm

This algorithm generates rules for all pairs of `sepalLength` and `sepalWidth` values within the iris dataset, offering a straightforward approach to nominal data classification.

[notebook](classification_nominal_data/one_r_algorithm.ipynb.ipynb)

### Logistic Regression

This classifier models the probability of categorical outcomes, making it suitable for nominal data analysis, such as the implemented for the iris dataset.

[notebook](classification_nominal_data/logistic_regression_classifier.ipynb)

### Decision Tree Classification

This section presents decision tree classifiers for both the iris and adult datasets, showcasing their versatility in nominal data classification. Decision trees provide intuitive visualizations of decision-making processes and can handle complex interactions between features.
[notebook](classification_nominal_data/decision_tree_classifier_adult_dataset.ipynb)
[notebook](classification_nominal_data/decision_tree_classifier_iris_dataset.ipynb)

## Clustering

### Agglomerative Clustering

Agglomerative clustering is a hierarchical clustering technique where each data point starts as its own cluster and then merges with other clusters based on some similarity metric. In the notebook you will view the implementation of this algorithm and the analysis of its metrics.

[notebook](clustering/agglomerative-clustering.ipynb)

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed, marking points as outliers if they lie alone in low-density regions. In the notebook you will view the implementation of this algorithm and the analysis of its metrics.

[notebook](clustering/dbscan-clustering.ipynb)

### K-means Clustering

K-means clustering is a popular partitioning clustering algorithm. It randomly initializes centroids and iteratively assigns data points to the nearest centroid, then updates centroids based on the mean of all points assigned to them. 

In this notebook, you can explore an implementation of the k-means clustering algorithm from scratch. Additionally, you can compare its results with the k-means clustering algorithm implemented in scikit-learn. Evaluation metrics such as Calinski-Harabaz index, Silhouette score, and Within Cluster Similarity are computed for comparison.

[notebook](clustering/k-means-clustering.ipynb)