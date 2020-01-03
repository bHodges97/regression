import numpy as np
from scipy.spatial import cKDTree

class knn_regressor:
    def __init__(self):
        pass

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


