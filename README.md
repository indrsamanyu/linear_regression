# Linear Regression From Scratch

A simple implementation of **Linear Regression from scratch using Python and NumPy**, without using machine learning libraries such as Scikit-learn.

## Overview

This project implements two optimization algorithms to learn the parameters (\(\theta\)) of a linear regression model:

* **Batch Gradient Descent (BGD)**
* **Stochastic Gradient Descent (SGD)**

The model predicts house prices based on:

* House size (1000 sq ft)
* Number of bedrooms

The hypothesis function is:

$$
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2
$$

## Gradient Descent Algorithms

### 1. Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the entire training dataset before updating the model parameters.

**Update rule:**

$$
\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

Where:

* \(\alpha\) is the learning rate.
* \(m\) is the number of training examples.
* \(x_j^{(i)}\) is the j-th feature of the i-th training example.
* \(y^{(i)}\) is the actual output.

**How it works:**

1. Calculate predictions for the entire training dataset.
2. Calculate the errors and gradients.
3. Update all model parameters simultaneously.
4. Repeat for the specified number of epochs.

### 2. Stochastic Gradient Descent

Stochastic Gradient Descent updates the model parameters after processing each individual training example.

**Update rule:**

$$
\theta_j := \theta_j - \alpha(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

**How it works:**

1. Iterate through the training examples.
2. Calculate the prediction and error for each example.
3. Immediately update the model parameters.
4. Repeat the process for the specified number of epochs.

Unlike Batch Gradient Descent, SGD does not wait for the entire dataset to calculate a gradient before updating the parameters.

## BGD vs SGD

| Feature                | Batch Gradient Descent | Stochastic Gradient Descent |
| ---------------------- | ---------------------- | --------------------------- |
| Gradient calculation   | Entire dataset         | One example                 |
| Parameter updates      | Once per epoch         | Once per example            |
| Convergence            | Generally smoother     | More fluctuating            |
| Computation per update | Higher                 | Lower                       |
| Memory requirements    | Can be higher          | Generally lower             |
| Training               | More stable updates    | More frequent updates       |

## Features

* Manual implementation of Batch Gradient Descent and Stochastic Gradient Descent
* Bias term included in the model
* Train/test data split
* Multiple training epochs
* Prediction on unseen data
* 3D visualization of training data and the regression plane using Matplotlib

## Tech Stack

* Python
* NumPy
* Matplotlib

## Visualization

The project uses Matplotlib to create a 3D visualization of the training data and the regression plane learned by the model.

The visualization helps illustrate how the linear regression model fits the data using two input features.

## Purpose

This project was built to understand the **mathematics and internal working of linear regression and gradient descent** rather than relying on pre-built machine learning libraries.

Implementing both Batch and Stochastic Gradient Descent provides a practical understanding of how different optimization algorithms learn model parameters.
