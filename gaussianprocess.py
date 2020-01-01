import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class gaussian_process_regressor:
    def __init__(self):
        pass


    def train(self,X,y,T):
        #TODO: prior is mean of training set or normalise to 0???
        var_y = np.var(y)

        K = rbf(X, X) + var_y * np.identity(y.size)
        Kx = rbf(X, T)
        Kxx = rbf(T, T)
        inv = np.linalg.inv(K)
        e = np.dot(np.dot(Kx.T,inv),y)
        return e



    def skgp(X,y,T):
        gpr = GaussianProcessRegressor().fit(X, y)
        return gpr.predict(T)
def sumsquared(x):
    return np.einsum('ij,ij->i',x,x)

def rbf(x1,x2,r2 = 2):
    norm = sumsquared(x1)[:,None] + sumsquared(x2) - 2 * np.dot(x1, x2.T)
    return np.exp( -norm  / r2)

