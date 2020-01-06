import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from numpy.linalg import cholesky, inv, pinv, lstsq
from scipy.optimize import fmin_l_bfgs_b
from utils import *

class gaussian_process_regressor:
    def fit(self,X,y, noise=0.1):
        #TODO: prior is mean of training set or normalise to 0???
        self.X = X
        self.y = y
        self.noise = noise

        self.K = rbf(X,X) # + sig_noise * np.identity(y.size)
        self.L = cholesky(self.K)
        self.L_inv = inv(self.L)
        self.a = self.L_inv.T.dot(self.L_inv.dot(y))

    def predict(self,T):
        X,y = self.X,self.y
        Kx = rbf(X, T)
        Kxx = rbf(T, T)
        #inv = inv(K)
        #self.mu = Kx.T.dot(inv).dot(y)
        #self.cov = Kxx - Kx.T.dot(inv).dot(Kx)
        self.mu = Kx.T.dot(self.a)
        v = self.L_inv.dot(Kx)
        self.cov = Kxx - v.T.dot(v)
        self.optimize()
        return self.mu

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

    def optimize(self):
        X,y = self.X, self.y
        n = X.shape[0]
        norm = sumsquared(X)[:,None] + sumsquared(X)[None,:] - 2 * np.dot(X,X.T)
        noise = self.noise * np.eye(n)

        def rbf_kernel(l):
            return np.exp( -norm  / (2*l))

        def log_marginal(*params):
            K = rbf_kernel(*params) + noise
            return 0.5 * (-y.T.dot(inv(K)).dot(y) - np.log(K) - n * np.log(2*np.pi))

        def rbf_1(l,d=1):
            nl =  norm / l**2
            return d * np.exp(nl/2).dot(-nl)

        def init():
            guess = [1]
            bounds = [(0,None)]
            res = fmin_l_bfgs_b(log_marginal,guess,rbf_1,bounds=bounds)
            print(res)
            #o = log_marginal(1)
            #print(o)

        init()


def kernel(x1, x2, *params):
    return self.rbf(x1,x2, *params)

def rbf(x1,x2, l = 0.01):
    norm = sumsquared(x1)[:,None] + sumsquared(x2)[None,:] - 2 * np.dot(x1, x2.T)
    return np.exp( -norm  / (2*l))
