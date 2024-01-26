# %% [markdown]
# # Logistic Regression

# %% [markdown]
# ### Objective Function
# 
# The objective function is $Q (w; X, y) = \frac{1}{n} \sum_{i=1}^n \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $.
# 
# When $\lambda = 0$, the model is a regular logistic regression and when $\lambda > 0$, it essentially becomes a regularized logistic regression.

# %%
import numpy as np
import pandas as pd
import math

# %%
# Inputs:
#     w: weight: d-by-1 matrix
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: regularization parameter: scalar
# Return:
#     objective function value, or loss (scalar)
def objective(w, x, y, lam):
    q=0
    n, d = x.shape
    for i in range(0,n):
        xi = x[i]
        xw = numpy.dot(x[i],w)
        
        exped = numpy.exp(y[i]*xw)
#        
        q = q+ numpy.log(1+(1/exped))  + (lam/2)*(numpy.linalg.norm(w,ord=2))
        
    return (1/n)*q


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# %% [markdown]
# ### Gradient Descent
# The gradient at $w$ for regularized logistic regression is  $g = - \frac{1}{n} \sum_{i=1}^n \frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$

# %%
# Inputs:
#     w: weight: d-by-1 matrix
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: regularization parameter: scalar
# Return:
#     g: gradient: d-by-1 matrix

def gradient(w, x, y, lam):
    g=0
    n, d = x.shape
    for i in range(n):
        numerator = np.dot(x[i],y[i])
        xw = numpy.dot(x[i],w)
        denominator = 1 + np.exp(y[i]*xw)
        g = g+ numerator/denominator + lam*w
    return -((1/n)*g) + lam*w

# %%
# Inputs:
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     learning_rate: scalar
#     w: weights: d-by-1 matrix, initialization of w
#     max_epoch: integer, the maximal epochs
# Return:
#     w: weights: d-by-1 matrix, the solution
#     objvals: a record of each epoch's objective value

def gradient_descent(x, y, lam, learning_rate, w, max_epoch=100):
    history = np.zeros((max_epoch, 1))
    for i in range(max_epoch):
        w = w-(learning_rate)*gradient(w,x,y,lam)
        history[i] = objective(w,x,y,lam)
    return w,history

# %% [markdown]
# ### Stochastic Gradient Descent
# 
# New objective function $Q_i (w) = \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $. 
# 
# The stochastic gradient at $w$ is $g_i = \frac{\partial Q_i }{ \partial w} = -\frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.

# %%
# Inputs:
#     w: weights: d-by-1 matrix
#     xi: data: 1-by-d matrix
#     yi: label: scalar
#     lam: scalar, the regularization parameter
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i
def stochastic_objective_gradient(w, xi, yi, lam):
    
    q=0
    xw = numpy.dot(xi,w)
    exped = numpy.exp(yi*xw)
    q = numpy.log(1+(1/exped))  + (lam/2)*(numpy.linalg.norm(w,ord=2))
    
    numerator = np.dot(xi,yi)
    xw = numpy.dot(xi,w)
    denominator = 1 + np.exp(yi*xw)
    g = -numerator/denominator + lam*w
    
    return g, q

# %%
# Inputs:
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     learning_rate: scalar
#     w: weights: d-by-1 matrix, initialization of w
#     max_epoch: integer, the maximal epochs
# Return:
#     
#     w: weights: d-by-1 matrix, the solution
#     objvals: a record of each epoch's objective value
#     Record one objective value per epoch (not per iteration)

def sgd(x, y, lam, learning_rate, w, max_epoch=100):
    history = np.zeros((max_epoch, 1))
    n = len(y)
    for i in range(max_epoch):
        obj=0
        permutation = np.random.permutation(n)
        x_permuted = x[permutation, :]
        y_permuted = y[permutation]
        for j in range(n):
            gradient, objective = stochastic_objective_gradient(w,x[j],y[j],lam)
            w = w-(learning_rate)*gradient
            obj += objective
        history[i] = obj/n
    return w, history


# %% [markdown]
# ### Mini-Batch Gradient Descent
# 
# $Q_I (w) = \frac{1}{b} \sum_{i \in I} \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $, where $I$ is a set containing $b$ indices randomly drawn from $\{ 1, \cdots , n \}$ without replacement.
# 
# The stochastic gradient at $w$ is $g_I = \frac{\partial Q_I }{ \partial w} = \frac{1}{b} \sum_{i \in I} \frac{- y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.

# %%
# Inputs:
#     w: weights: d-by-1 vector
#     xi: data: b-by-d matrix
#     yi: label: scalar
#     lam: scalar, the regularization parameter
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i

def mb_objective_gradient(w, xi, yi, lam):
    q=0
    b, d = xi.shape
    for i in range(0,b):
        xw = numpy.dot(xi[i],w)
        exped = numpy.exp(yi[i]*xw)
        q = q+ numpy.log(1+(1/exped))  + (lam/2)*(numpy.linalg.norm(w,ord=2))
        
    g=0
    b, d = xi.shape
    for j in range(b):
        numerator = np.dot(xi[j],yi[j])
        xw = numpy.dot(xi[j],w)
        denominator = 1 + np.exp(yi[j]*xw)
        g = g+ numerator/denominator + lam*w
        
    return (1/b)*q , -((1/b)*g) + lam*w  

# %%
# Inputs:
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     learning_rate: scalar
#     w: weights: d-by-1 matrix, initialization of w
#     max_epoch: integer, the maximal epochs
# Return:
#     w: weights: d-by-1 matrix, the solution
#     objvals: a record of each epoch's objective value
#     Record one objective value per epoch (not per iteration)

def mbgd(x, y, lam, learning_rate, w, max_epoch=100):
    batch_size = 20
    n = len(y)
    history = np.zeros((max_epoch, 1))
    for i in range(max_epoch):
        obj_av = 0
        permutation = np.random.permutation(n)
        X_permuted = x[permutation, :]
        y_permuted = y[permutation]
        for j in range(0, n, batch_size):
            X_batch = X_permuted[j:j+batch_size, :]
            y_batch = y_permuted[j:j+batch_size]
            objective, gradient = mb_objective_gradient(w, X_batch, y_batch, lam)
            w = w - learning_rate * gradient
            obj_av+= objective
        history[i] = obj_av/(n/batch_size)
    return w, history   


