# Logistic Regression

This module provides an implementation of logistic regression, a machine learning algorithm used for binary classification problems. It includes functions for calculating the objective function, gradient, and performing gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent (MBGD).

## Functions

1. `objective(w, x, y, lam)`: Calculates the objective function for logistic regression.

$Q (w; X, y) = \frac{1}{n} \sum_{i=1}^n \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $.

2. `gradient(w, x, y, lam)`: Calculates the gradient of the objective function.

The gradient at $w$ for regularized logistic regression is  $g = - \frac{1}{n} \sum_{i=1}^n \frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$

3. `gradient_descent(x, y, lam, learning_rate, w, max_epoch=100)`: Performs gradient descent to optimize the objective function.

4. `stochastic_objective_gradient(w, xi, yi, lam)`: Calculates the stochastic objective gradient.

The objective function for SGD is $Q_i (w) = \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $.

5. `sgd(x, y, lam, learning_rate, w, max_epoch=100)`: Performs stochastic gradient descent to optimize the objective function.

The stochastic gradient at $w$ is $g_i = \frac{\partial Q_i }{ \partial w} = -\frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.

6. `mb_objective_gradient(w, xi, yi, lam)`: Calculates the mini-batch objective gradient.

The ojective function for MBGD is $Q_I (w) = \frac{1}{b} \sum_{i \in I} \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $, where $I$ is a set containing $b$ indices randomly drawn from $\{ 1, \cdots , n \}$ without replacement.

7. `mbgd(x, y, lam, learning_rate, w, max_epoch=100)`: Performs mini-batch gradient descent to optimize the objective function.

The stochastic gradient at $w$ is $g_I = \frac{\partial Q_I }{ \partial w} = \frac{1}{b} \sum_{i \in I} \frac{- y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.

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


