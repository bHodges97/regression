import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.gaussian_process import GaussianProcessRegressor
from timeit import timeit
from numpy.linalg import cholesky
from utils import *



class gaussian_process_regressor:
    def __init__(self):
        pass


    def train(self,X,y):
        #TODO: prior is mean of training set or normalise to 0???
        self.X = X
        self.y = y
        self.K = rbf(X,X) # + sig_noise * np.identity(y.size)
        self.L = np.linalg.cholesky(K)
        self.L_inv = np.linalg.inv(self.L)
        self.a = self.L_inv.T.dot(self.L_inv.dot(y))

    def fit(self,T):
        Kx = rbf(X, T)
        Kxx = rbf(T, T)
        #inv = np.linalg.inv(K)
        #self.mu = Kx.T.dot(inv).dot(y)
        #self.cov = Kxx - Kx.T.dot(inv).dot(Kx)
        self.mu = Kx.T.dot(self.a)
        v = self.L_inv.dot(Kx)
        self.cov = Kxx - v.T.dot(v)
        return self.mu

    def skgp(X,y,T):
        gpr = GaussianProcessRegressor().fit(X, y)
        return gpr.predict(T)

    def plot(self):
        X = self.T.ravel()
        mu = self.mu.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(self.cov))

        o = np.argsort(X)
        X = X[o]
        mu = mu[o]
        plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
        plt.plot(X, mu, label='Mean')
        plt.plot(self.X, self.y, 'rx')
        plt.legend()
        plt.show()


def rbf(x1,x2, l = 0.01):
    norm = sumsquared(x1)[:,None] + sumsquared(x2) - 2 * np.dot(x1, x2.T)
    return np.exp( -norm  / (2*l))

if __name__ == "__main__":
    #x_train = np.arange(-3, 4, 1).reshape(-1, 1)
    x_train = np.arange(-5,5,0.1).reshape(-1, 1)
    y_train = np.sin(x_train)# + noise * np.random.randn(*x_train.shape)
    #y_train = 3 * x_train + 1
    x_test = np.arange(-5, 5, 0.2).reshape(-1, 1)

    gaussian = gaussian_process_regressor()
    y_pred = gaussian.alg2(x_train,y_train,x_test)
    time = 0
    #time = timeit(lambda :gaussian.alg2(x_train,y_train,x_test),number=5000)
    print("done",time)
    gaussian.plot()

