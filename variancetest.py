import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('sarcos_inv.csv', delimiter=',')
X = data[:,:-1]
y = data[:,-1]

from timeit import timeit
import time


#X = np.arange(10).reshape(-1,1)
#y = np.sin(X).ravel()

def oldway(feature):
    best = np.inf,
    x = X[:,feature]
    print(feature,x.shape)
    b = []
    for split in np.unique(x)[:-1]:
        #split left and right
        mask = x <= split
        #get variance * size
        ln = np.count_nonzero(mask)
        rn = mask.size - ln
        var = ln * np.var(y[mask]) + rn * np.var(y[~mask])
        print(rn*np.var(y[~mask]),np.sum(y[~mask]))
        b.append(var)
        if var < best[0]: #minimise weighted variance
            best = (var,mask,split,feature)
    print(best)
    print(np.array(b).T)


def cumulative_mse(y, y_n, y_squared, reverse = False):
    if reverse:
        y = np.flipud(y)
        y_squared = np.flipud(y_squared)
    cum_squared = np.cumsum(y,axis=0)**2
    squared_cum = np.cumsum(y_squared,axis=0)
    return squared_cum - cum_squared / y_n

def findsplit(X,y,features):
    x_order = np.argsort(X[:,features],axis=0)
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
    return best_split, features[best_feature]


def findsplitonline(X,y,features):
    x_order = np.argsort(X[:,features],axis=0)
    y_n = np.arange(1,X.shape[0]+1,dtype=np.int).reshape(-1,1)
    y_l = np.empty(x_order.shape,dtype=y.dtype)

    for idx,x in enumerate(x_order.T):
        y_l[:,idx] = y[x]

    mean_r = np.mean(y_l,axis=0).reshape(-1)
    mse_r = (np.var(y_l,axis=0) * (y.shape[0])).reshape(-1)
    mean_l = np.zeros_like(mean_r)
    mse_l = np.zeros_like(mean_r)
    row_i = np.zeros_like(mean_r)
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
    return best_split,features[best_feature]
#print(out)

t0 = time.time()
#out = findsplit(X,y,np.arange(X.shape[1],dtype=np.int))
out = findsplitonline(X,y,[5,7,8,9])
print(out)
print(time.time()-t0)
