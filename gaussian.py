import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from numpy.linalg import cholesky, inv, pinv, lstsq, det
from scipy.optimize import fmin_l_bfgs_b
from utils import *

class gaussian_process_regressor:
    def fit(self,X,y, noise=0.1):
        self.X = X
        self.y = y
        if X.shape[0] > 200:
            print("truncating")
            size = int(X.shape[0] * 0.15)
            size = 200
            self.X = X[:size]
            self.y = y[:size]

        #TODO: prior is mean of training set or normalise to 0???
        self.noise = noise
        print("learning")
        t0 = time.time()
        self.params = self.optimize()
        print(time.time()-t0)
        #self.X = X[:size]
        #self.y = y[:size]

        self.K = rbf(X,X, *self.params)  + noise * np.identity(y.size)
        self.L = cholesky(self.K)
        self.L_inv = inv(self.L)
        self.a = self.L_inv.T.dot(self.L_inv.dot(y))

    def predict(self,T):
        self.T = T
        X,y = self.X,self.y
        Kx = rbf(X, T, *self.params)
        Kxx = rbf(T, T, *self.params)
        #inv = inv(K)
        #self.mu = Kx.T.dot(inv).dot(y)
        #self.cov = Kxx - Kx.T.dot(inv).dot(Kx)
        self.mu = Kx.T.dot(self.a)
        v = self.L_inv.dot(Kx)
        self.cov = Kxx - v.T.dot(v)
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
        X,y = self.X, self.y#.reshape(-1,1)
        n = X.shape[0]
        norm = sumsquared(X)[:,None] + sumsquared(X)[None,:] - 2 * np.dot(X,X.T)
        noise = self.noise * np.eye(n)

        def rbf_kernel(d,l):
            return d * np.exp( -norm  / l)

        def log_marginal(params):
            d,l = params[0],params[1]
            K = rbf_kernel(d,l) + noise
            #0.5 * ln |K| - o,5 y.t * K^-1 * y - N/2 ln(2pi)
            return -0.5 * (-y.T.dot(inv(K)).dot(y) - np.log(det(K)) - n * np.log(2*np.pi))

        def grad_log(params):
            d,l = params[0],params[1]
            #l = np.exp(l)
            #d = np.exp(2*d)
            dkdl = - d * np.exp(-norm/2).dot(norm) / l**2
            dkdd = np.exp(-norm/l)

            K = rbf_kernel(l,d) + noise
            invk = inv(K)
            a = invk.dot(y.T)

            #ddl = 0.5 * np.trace((a.dot(a.T)- invk).dot(dkdl))
            #ddd = 0.5 * np.trace((a.dot(a.T)- invk).dot(dkdd))

            ddl = 0.5 * y.T.dot(invk).dot(dkdl).dot(a) - 0.5*np.trace(invk.dot(dkdl))
            ddd = 0.5 * y.T.dot(invk).dot(dkdd).dot(a) - 0.5*np.trace(invk.dot(dkdd))

            o = -np.array([ddd,ddl])
            return o

        def init():
            guess = np.array([1,0.01])
            bounds = [(1e5,None),(1e5,None)]
            #res = fmin_l_bfgs_b(log_marginal,guess,grad_log,bounds=bounds)[0]
            res = sp.optimize.minimize(log_marginal,guess,method='L-BFGS-B',bounds=bounds)#['x']
            print(res)
            return res['x']

        return list(init())


def kernel(x1, x2, *params):
    return self.rbf(x1,x2, *params)

def rbf(x1,x2,d=1, l = 0.01):
    norm = sumsquared(x1)[:,None] + sumsquared(x2)[None,:] - 2 * np.dot(x1, x2.T)
    return d * np.exp( -norm  / l)
