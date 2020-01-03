import numpy as np
from multiprocessing.pool import Pool
from itertools import repeat
from utils import *

class random_forest_regressor:
    def __init__(self,m=20):
        self.m = m
        self.trees = []

    def train(self,X,y):
        k = int(np.ceil(np.sqrt(X.shape[1])))
        #select random samples for bagging
        samples = [np.random.choice(X.shape[0],X.shape[0]) for _ in range(self.m)]
        #select random subspace
        args = [(X[sample],y[sample],k) for sample in samples]
        pool = Pool(processes=4)
        self.trees = list(pool.starmap(build_tree, args))
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

def kfeatures(X,k):
    if k == None:
        return np.arange(X.shape[1],dtype=np.int)
    return np.sort(np.random.choice(X.shape[1],k,replace=False))

def build_tree(X, y, k = None, minsplit = 2, converge = 1e-6, mse = np.inf,  max_depth = np.inf):
    if max_depth == 0 or y.size <= minsplit:
        if y.size == 0:
            print("oh no")
        return np.mean(y),

    X = ensure2D(X)
    subspace = kfeatures(X,k)
    feature,split,new_mse = findsplit(X[:,subspace],y)

    if split is None or  mse - new_mse < converge:# No splits could be made
        return np.mean(y),

    feature = subspace[feature]
    mask = X[:,feature] <= split
    lefty = y[mask]
    righty = y[~mask]
    if lefty.size == 0 or righty.size == 0:
        return np.mean(y),

    args = (k, minsplit, converge, new_mse,  max_depth - 1)
    left = build_tree(X[mask], lefty, *args)
    right = build_tree(X[~mask], righty, *args)
    return (feature, split,left,right)

def findsplit(X,y):
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


def findsplit_cummulative(X,y):
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


