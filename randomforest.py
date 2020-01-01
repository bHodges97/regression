import numpy as np
from multiprocessing.pool import Pool
import time

class random_forest_regressor:
    def __init__(self,m=100):
        self.m = m
        self.trees = []

    def train(self,X,y):
        k = int(np.ceil(np.sqrt(X.shape[1])))
        threads = []

        t0 = time.time()
        samples = [np.random.choice(X.shape[0],X.shape[0]) for _ in range(self.m)]
        args = [(X[sample,np.random.choice(k,k,replace=False)],y[sample]) for sample in samples]
        pool = Pool(processes=4)
        self.trees = list(pool.imap_unordered(build_tree, args))
        #self.trees = list(map(build_tree,args))
        t1 = time.time()
        print("time",t1-t0)

    def predict(self,X):
        prediction = np.zeros(X.shape[0])
        for idx,x in enumerate(X):
            for tree in self.trees:
                while len(tree) > 1:
                    if x[tree[0]] <= tree[1]:
                        tree = tree[2]
                    else:
                        tree = tree[3]
                prediction[idx] += tree[0]
        return prediction / self.m

def build_tree(train, max_depth = np.inf):
    X,y = train
    if max_depth == 0 or y.size == 1:
        return y[0],

    if X.ndim == 1:#ensure 2d for toy problem
        X = X[:, None]
    best = np.inf,
    for feature, x in enumerate(X.T):
        best = np.inf,
        for split in np.unique(x)[:-1]:
            #split left and right
            mask = x <= split
            #get variance * size
            ln = np.count_nonzero(mask)
            rn = mask.size - ln
            var = ln * np.var(y[mask]) + rn * np.var(y[~mask])
            if var < best[0]: #minimise weighted variance
                best = (var,mask,split,feature)
    if len(best) == 1:# No splits could be made
        return np.mean(y),
    var, mask, split, feature = best
    left = build_tree((X[mask,:],y[mask]),max_depth-1)
    right = build_tree((X[~mask,:],y[~mask]),max_depth-1)
    return (feature, split,left,right)

