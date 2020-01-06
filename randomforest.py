import numpy as np
from multiprocessing.pool import Pool
from utils import *

class random_forest_regressor:
    def __init__(self,m=1,k=None):
        self.m = m
        self.trees = []
        self.k = k

    def fit(self,X,y):
        k = self.k
        if k is None:
            k = X.shape[1] // 2
        #k = 5#np.ceil(X.shape[1]**0.5)//1# // 2 + 1

        #select random samples for bagging
        samples = (np.random.choice(X.shape[0],X.shape[0]) for _ in range(self.m))
        #select random subspace
        args = ((X[sample],y[sample],k) for sample in samples)
        pool = Pool(processes=4)
        self.trees = list(pool.starmap(self._build_tree, args))
        #self.trees = list(map(build_tree,args))

    def predict(self,X):
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

    def _build_tree(self, X, y, k = None, minsplit = 2, converge = 1e-6, mse = np.inf,  max_depth = np.inf):
        if max_depth == 0 or y.size <= minsplit:
            if y.size == 0:
                print("oh no")
            return np.mean(y),

        X = ensure2D(X)
        subspace = _kfeatures(X,k)
        feature,split,new_mse = _findsplit_cummulative(X[:,subspace],y)

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

def _kfeatures(X, k):
    if k == None:
        return np.arange(X.shape[1],dtype=np.int)
    return np.sort(np.random.choice(X.shape[1],k,replace=False))

def _findsplit(X,y):
    x_order = np.argsort(X,axis=0)
    y_n = np.arange(1,X.shape[0]+1,dtype=np.int).reshape(-1,1)
    y_repeat =np.repeat(y.reshape(-1,1),X.shape[1],axis=1)
    y_l = np.take_along_axis(y_repeat, x_order, axis=0)
    x_sorted = np.take_along_axis(X, x_order, axis=0)
    big_mask = x_sorted[:-1] == x_sorted[1:]
    mse = np.empty(big_mask.shape)

    mse_r = (np.var(y_l,axis=0) * y.shape[0])
    mean_r = np.mean(y_l,axis=0)
    mean_l = np.zeros_like(mean_r)
    mse_l = np.zeros_like(mean_r)
    total = y_l.shape[0]

    for idx,row in enumerate(y_l[:-1]):
        delta_l = row - mean_l
        mean_l += delta_l / (idx+1)
        mse_l += delta_l * (row - mean_l)
        delta_r = row - mean_r
        mean_r -= delta_r / (total - (idx +1))
        mse_r -= delta_r * (row - mean_r)
        mse[idx] = mse_l + mse_r

    mse = np.ma.masked_array(mse,big_mask)
    split,best_feature = np.unravel_index(np.argmin(mse),big_mask.shape)
    best_mse = mse[split,best_feature]
    best_split = x_sorted[split,best_feature]#best_splits[best_feature]

    return best_feature, best_split, best_mse

def _findsplit_cummulative(X, y):
    x_order = np.argsort(X,axis=0)
    y_n = np.arange(1,X.shape[0]+1,dtype=np.int).reshape(-1,1)
    y_repeat = np.repeat(y.reshape(-1,1),X.shape[1],axis=1)
    y_l = np.take_along_axis(y_repeat, x_order, axis=0)
    x_sorted = np.take_along_axis(X, x_order, axis=0)
    big_mask = x_sorted[:-1] == x_sorted[1:]

    y_l_squared = y_l ** 2
    lmse = cumulative_mse(y_l,y_n,y_l_squared)[:-1]
    rmse = cumulative_mse(y_l,y_n,y_l_squared, reverse = True)[:-1]
    mse = lmse + np.flipud(rmse)
    split,best_feature = np.unravel_index(np.argmin(mse),big_mask.shape)
    best_mse = mse[split,best_feature]
    best_split = x_sorted[split,best_feature]

    return best_feature, best_split, best_mse
