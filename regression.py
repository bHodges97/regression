import numpy as np
import scipy as sp
from scipy.spatial import cKDTree

from utils import *

class linear_regressor:
    def __init__(self, method = "pinv"):
        if method == "pinv": #pseudo inverse
            self.solver = lambda X,y: pinv(X).dot(y)
        elif method == "inv":
            self.solver = lambda X,y: inv(X.T.dot(X)).dot(X.T).dot(y)
        elif method == "lstsq":
            self.solver = lambda X,y: lstsq(X,y,rcond=None)[0]

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, T):
        X_mean = np.mean(self.X,axis=0) #shift intercept
        y_mean = np.mean(self.y)
        X = self.X - X_mean
        y = self.y - y_mean
        b = self.solver(X,y)
        e = y_mean - X_mean.dot(b) #y = bx+c
        return T.dot(b) + e

class knn_regressor:
    def fit(self,X,y):
        self.kdtree = cKDTree(X)
        self.X = X
        self.y = y

    def predict(self, T, k=5):
        #kdtree
        neighbours = self.kdtree.query(T,k=k)[1]
        if k == 1:
            return self.y.take(neighbours)
        return np.mean(self.y.take(neighbours),axis=1)

    def brutepredict(self, T, k=5):
        #brute force solution
        X,y = self.X, self.y
        def knn_func(xpred,k):
            distances = np.sum((X-xpred)**2,axis=1)
            indices = np.argsort(distances)[:k]
            return np.mean(y[indices])
        knn_vect = np.vectorize(knn_func)
        return knn_vect(T,k).ravel()
