import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
from scipy.spatial import cKDTree
from itertools import repeat
from multiprocessing.pool import Pool

from utils import *

class linear_regressor:
    def __init__(self, method = "pinv"):
        if method == "pinv": #pseudo inverse
            self.solver = lambda X,y: np.linalg.pinv(X).dot(y)
        elif method == "inv":
            self.solver = lambda X,y: np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        elif method == "lstsq":
            self.solver = lambda X,y: np.linalg.lstsq(X,y,rcond=None)[0]

    def train(self, X, y):
        self.X = X
        self.y = y

    def fit(self, T):
        X_mean = np.mean(self.X,axis=0) #shift intercept
        y_mean = np.mean(self.y)
        X = self.X - X_mean
        y = self.y - y_mean
        b = self.solver(X,y)
        e = y_mean - X_mean.dot(b) #y = bx+c
        return T.dot(b) + e

class knn_regressor:

    def train(self,X,y):
        self.tree = cKDTree(X)
        self.X = x
        self.y = y

    def fit(self, T, k=5):
        #kdtree
        neighbours = self.kdtree.query(T,k=k)[1]
        if k == 1:
            return self.y.take(neighbours)
        return np.mean(self.y.take(neighbours),axis=1)

    def brutefit(self, T, k=5):
        #brute force solution
        X,y = self.X, self.y
        def knn_func(xpred,k):
            distances = np.sum((X-xpred)**2,axis=1)
            indices = np.argsort(distances)[:k]
            return np.mean(y[indices])
        knn_vect = np.vectorize(knn_func)
        return  = knn_vect(T,k).ravel()


class gaussian_process_regressor:

    def train(self,X,y):
        #TODO: prior is mean of training set or normalise to 0???
        self.X = X
        self.y = y
        self.K = self.rbf(X,X) # + sig_noise * np.identity(y.size)
        self.L = np.linalg.cholesky(K)
        self.L_inv = np.linalg.inv(self.L)
        self.a = self.L_inv.T.dot(self.L_inv.dot(y))

    def fit(self,T):
        Kx = self.rbf(X, T)
        Kxx = self.rbf(T, T)
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

    @staticmethod
    def rbf(x1,x2, l = 0.01):
        norm = sumsquared(x1)[:,None] + sumsquared(x2) - 2 * np.dot(x1, x2.T)
        return np.exp( -norm  / (2*l))

class random_forest_regressor:
    def __init__(self,m=10):
        self.m = m
        self.trees = []

    def train(self,X,y):
        k = X.shape[1] // 2

        #select random samples for bagging
        samples = [np.random.choice(X.shape[0],X.shape[0]) for _ in range(self.m)]
        #select random subspace
        args = [(X[sample],y[sample],k) for sample in samples]
        pool = Pool(processes=4)
        self.trees = list(pool.starmap(self._build_tree, args))
        #self.trees = list(map(build_tree,args))

    def fit(self,X):
        prediction = np.zeros(X.shape[0])
        for idx,x in enumerate(X):
            for tree in self.trees:
                while len(tree) > 1:
                    #if feature <= split
                    if x[tree[0]] <= tree[1]:
                        tree = tree[2]
                    else:
                        tree = tree[3]
                prediction[idx] += tree[0]
            prediction[idx] /= self.m
        return prediction

    @staticmethod
    def _kfeatures(X, k):
        if k == None:
            return np.arange(X.shape[1],dtype=np.int)
        return np.sort(np.random.choice(X.shape[1],k,replace=False))

    @staticmethod
    def _build_tree(X, y, k = None, minsplit = 2, converge = 1e-6, mse = np.inf,  max_depth = np.inf):
        if max_depth == 0 or y.size <= minsplit:
            if y.size == 0:
                print("oh no")
            return np.mean(y),

        X = ensure2D(X)
        subspace = self._kfeatures(X,k)
        feature,split,new_mse = _findsplit(X[:,subspace],y)

        if split is None or  mse - new_mse < converge:# No splits could be made
            return np.mean(y),

        feature = subspace[feature]
        mask = X[:,feature] <= split
        lefty = y[mask]
        righty = y[~mask]
        if lefty.size == 0 or righty.size == 0:
            return np.mean(y),

        args = (k, minsplit, converge, new_mse,  max_depth - 1)
        left = self._build_tree(X[mask], lefty, *args)
        right = self._build_tree(X[~mask], righty, *args)
        return (feature, split,left,right)

    @staticmethod
    def _findsplit(X,y):
        x_order = np.argsort(X,axis=0)
        y_n = np.arange(1,X.shape[0]+1,dtype=np.int).reshape(-1,1)
        y_l = np.empty(x_order.shape,dtype=y.dtype)

        for idx,x in enumerate(x_order.T):
            y_l[:,idx] = y[x]

        y_mse = (np.var(y_l,axis=0) * (y.shape[0]))
        mean_r = np.mean(y_l,axis=0).reshape(-1)
        mse_r = y_mse.reshape(-1)
        mean_l = np.zeros_like(mean_r)
        mse_l = np.zeros_like(mean_r)
        total = y_l.shape[0]
        best_mse = np.inf
        best_feature = None
        best_split = None
        for idx,row in enumerate(y_l[:-1]):
            delta_l = row - mean_l
            mean_l += delta_l / (idx+1)
            mse_l += delta_l * (row - mean_l)
            delta_r = row - mean_r
            mean_r -= delta_r / (total - (idx +1))
            mse_r -= delta_r * (row - mean_r)
            mse = mse_l + mse_r

            splits = X[x_order[idx]]
            for feature, score in enumerate(mse):
                if score < best_mse:
                    split = X[x_order[idx,feature],feature]
                    if split != X[x_order[idx-1,feature],feature]:
                        best_mse = score
                        best_feature = feature
                        best_split = split
        return best_feature, best_split, best_mse

    @staticmethod
    def _findsplit_cummulative(X, y):
        x_order = np.argsort(X,axis=0)
        y_n = np.arange(1,X.shape[0]+1,dtype=np.int).reshape(-1,1)
        y_l = np.empty(x_order.shape,dtype=y.dtype)
        for idx,x in enumerate(x_order.T):
            y_l[:,idx] = y[x]
        y_l_squared = y_l ** 2
        lmse = cumulative_mse(y_l,y_n,y_l_squared)[:-1]
        rmse = cumulative_mse(y_l,y_n,y_l_squared, reverse = True)[:-1]
        mse = lmse + np.flipud(rmse)

        best_feature = None
        best_mse = np.inf
        best_split = None
        last_split = None
        for feature in range(x_order.shape[1]):
            for idx,x_split in enumerate(x_order[:-1,feature]):
                if X[x_split,feature] == X[x_order[idx+1,feature],feature]:
                    continue
                if mse[idx,feature] < best_mse:
                    best_split = X[x_split,feature]
                    best_mse = mse[idx,feature]
                    best_feature = feature
        return best_feature, best_split, best_mse


