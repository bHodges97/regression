import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from numpy.linalg import cholesky, inv, pinv, lstsq, det
from scipy.optimize import fmin_l_bfgs_b
from utils import *

class gaussian_process_regressor:
    def fit(self,X,y, noise=20):
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
        print("training time",time.time()-t0)
        #self.params = [10000,10000]
        self.X = X[500:4000]
        self.y = y[500:4000]

        self.K = rbf(self.X,self.X, *self.params)  + noise * np.identity(self.y.size)
        self.L = cholesky(self.K)
        self.L_inv = inv(self.L)
        self.a = self.L_inv.T.dot(self.L_inv.dot(self.y))

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
        if X.shape[1] > 1:
            print("No plotting 2d")
            return
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

        def log_p(params):
            d,l = params[0],params[1]
            K = rbf_kernel(d,l) + noise
            #L = cholesky(K)
            #L_inv = inv(L)
            #a = L_inv.T.dot(L_inv.dot(y))
            #0.5 * ln |K| - o,5 y.t * K^-1 * y - N/2 ln(2pi)
            logdet = np.linalg.slogdet(K)[1]
            #logdet = np.sum(np.log(np.diag(K)))
            log_p = 0.5 * (-y.T.dot(inv(K)).dot(y) - logdet - n * np.log(2*np.pi))
            return -log_p

        def grad_log(params):
            d,l = params[0],params[1]
            dkdl = - d * np.exp(-norm/2).dot(norm) / l**2
            dkdd = np.exp(-norm/l)

            K = rbf_kernel(d,l) + noise
            invk = inv(K)
            a = invk.dot(y.T)

            ddl = 0.5 * y.T.dot(invk).dot(dkdl).dot(a) - 0.5*np.trace(invk.dot(dkdl))
            ddd = 0.5 * y.T.dot(invk).dot(dkdd).dot(a) - 0.5*np.trace(invk.dot(dkdd))

            o = -np.array([ddd,ddl])

            print("x",o,d,l)
            return o

        def log_p2(params):
            d,l = params[0],params[1]
            K = rbf_kernel(d,l) + noise

            L = np.linalg.cholesky(K)
            beta = np.linalg.solve(L.T, np.linalg.solve(L,y))
            logp = - 0.5 * np.dot(y.T,beta) - np.sum(np.log(np.diag(L))) - \
                0.5 * n * np.log(2*np.pi)
            print(logp,d,l)
            return -logp

        def init():
            guess = np.array([150,1])
            bounds = [(1e-8,None),(1e-8,None)]
            #res = fmin_l_bfgs_b(log_p2,guess,grad_log,bounds=bounds)
            res = sp.optimize.minimize(log_p2,guess,method='L-BFGS-B',bounds=bounds)#
            print("best params",res)
            if 'x' in res:
                res= res['x']
            else:
                res = res[0]
            return res[0],res[1]

        return init()


def kernel(x1, x2, *params):
    return self.rbf(x1,x2, *params)

def rbf(x1,x2,d=1, l = 0.01):
    norm = sumsquared(x1)[:,None] + sumsquared(x2)[None,:] - 2 * np.dot(x1, x2.T)
    return d * np.exp( -norm  / l)
