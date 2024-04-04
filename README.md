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

### Linear Regression
Explore linear regression using Gradient Descent! We start by manually implementing Gradient Descent on the London borough profiles dataset to predict the maximum.

![Linear Regression](images/LinearRegression.png)

Feel free to explore and experiment with the provided notebooks! Happy mining!


## Nominal Data Classification

This section delves into various algorithms tailored for classifying nominal data, providing insights into their applications and effectiveness.

### OneR (1R) Algorithm

This algorithm generates rules for all pairs of `sepalLength` and `sepalWidth` values within the iris dataset, offering a straightforward approach to nominal data classification.

### Logistic Regression

This classifier models the probability of categorical outcomes, making it suitable for nominal data analysis, such as the implemented for the iris dataset.

### Decision Tree Classification

This section presents decision tree classifiers for both the iris and adult datasets, showcasing their versatility in nominal data classification. Decision trees provide intuitive visualizations of decision-making processes and can handle complex interactions between features.