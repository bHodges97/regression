import numpy as np

def ensure2D(X):
    if X.ndim == 1:#ensure 2d for toy problem
        X = X[:, None]
    return X

def sumsquared(x):
    if x.ndim == 1:#ensure 2d for toy problem
        return np.inner(x,x)
    return np.einsum('ij,ij->i',x,x)

def squared_error(x1,x2):
    return sumsquared(x1) + sumsquared(x2) - 2 * np.dot(x1, x2)

def mean_squared_error(x1,x2):
    return squared_error(x1,x2)/x1.size

def cumulative_mse(y, y_n, y_squared, reverse = False):
    if reverse:
        y = np.flipud(y)
        y_squared = np.flipud(y_squared)
    cum_squared = np.cumsum(y,axis=0)**2
    squared_cum = np.cumsum(y_squared,axis=0)
    return squared_cum - cum_squared / y_n

