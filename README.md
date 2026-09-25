# Linear Regression From Scratch

A simple implementation of **Linear Regression from scratch using Python and NumPy**, without using machine learning libraries such as Scikit-learn.

## Overview

This project implements **Batch Gradient Descent** to learn the parameters (θ) of a linear regression model.

The model predicts house prices based on:

- House size (1000 sq ft)
- Number of bedrooms

The hypothesis function is:

\[
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2
\]

The parameters are learned using Batch Gradient Descent:

\[
\theta_j := \theta_j - \alpha\frac{1}{m}
\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
\]

## Features

- Manual implementation of Batch Gradient Descent
- Bias term included in the input matrix
- Train/test data split
- Prediction on unseen data
- 3D visualization of the training data and regression plane using Matplotlib

## Tech Stack

- Python
- NumPy
- Matplotlib

## Visualization

The 3D plot shows the actual training data points and the regression plane learned by the model.

## Purpose

This project was built to understand the **mathematics and internal working of linear regression and gradient descent**, rather than relying on pre-built ML libraries.
