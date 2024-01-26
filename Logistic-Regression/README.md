# Logistic Regression

This library provides an implementation of logistic regression, a machine learning algorithm used for binary classification problems. It includes functions for calculating the objective function, gradient, and performing gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent (MBGD).

## Functions

1. `objective(w, x, y, lam)`: Calculates the objective function for logistic regression.
2. `sigmoid(z)`: Calculates the sigmoid of `z`.
3. `gradient(w, x, y, lam)`: Calculates the gradient of the objective function.
4. `gradient_descent(x, y, lam, learning_rate, w, max_epoch=100)`: Performs gradient descent to optimize the objective function.
5. `stochastic_objective_gradient(w, xi, yi, lam)`: Calculates the stochastic objective gradient.
6. `sgd(x, y, lam, learning_rate, w, max_epoch=100)`: Performs stochastic gradient descent to optimize the objective function.
7. `mb_objective_gradient(w, xi, yi, lam)`: Calculates the mini-batch objective gradient.
8. `mbgd(x, y, lam, learning_rate, w, max_epoch=100)`: Performs mini-batch gradient descent to optimize the objective function.

## Usage

First, import the necessary libraries:

```python
import numpy as np
import pandas as pd
x = ... # input data
y = ... # target data
lam = ... # regularization parameter
learning_rate = ... # learning rate
w = ... # initial weights
max_epoch = 100 # number of epochs
```
You can then call the gradient descent function as follows:

```python
w, history = gradient_descent(x, y, lam, learning_rate, w, max_epoch)
w, history = sgd(x, y, lam, learning_rate, w, max_epoch)
w, history = mbgd(x, y, lam, learning_rate, w, max_epoch)
```


