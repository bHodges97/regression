import itertools
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import time
from scipy.spatial import cKDTree
from sklearn import neighbors,linear_model

from randomforest import *
from gaussianprocess import *
from utils import *

def toy_dataset(size=100):
    np.random.seed(0)
    noise = np.random.normal(0,0.2,(size,1))
    x = np.random.uniform(0, 5,(size,1))
    x = np.sort(x)
    #y = np.sin(x)#.ravel()
    y = (x * 1.5 ) + 5
    #y += noise
    #y = 1/5*np.sin(x*np.pi*2*5) + 2*x
    return split_data(np.hstack([x,y]))

def load_dataset():
    data = np.genfromtxt('sarcos_inv.csv', delimiter=',')
    np.random.shuffle(data)
    return split_data(data)

def split_data(data):
    rows = data.shape[0] // 10 * 8
    x_train = data[:rows,:-1]
    y_train = data[:rows,-1].ravel()
    x_test = data[rows:,:-1]
    y_test = data[rows:,-1].ravel()
    return (x_train,y_train,x_test,y_test)

def visualise(x_test,y_test,y_pred):
    plt.scatter(x_test, y_test, color='darkorange', label='data' ,s=5)
    plt.scatter(x_test, y_pred, color='navy', label='prediction', s=1)
    plt.show()

def nearest_neighbour(X,y,T,k=5):
    #brute force solution
    #def knn_func(xpred,k):
    #    distances = np.sum((X-xpred)**2,axis=1)
    #    indices = np.argsort(distances)[:k]
    #    return np.mean(y[indices])
    #knn_vect = np.vectorize(knn_func)
    #return  = knn_vect(T,k).ravel()

    #kdtree
    kdtree = cKDTree(X)
    neighbours = kdtree.query(T,k=k)[1]
    if k == 1:
        return y.take(neighbours)
    return np.mean(y.take(neighbours),axis=1)

def linear_regression(X,y,T):
    X_mean = np.mean(X,axis=0) #shift intercept
    y_mean = np.mean(y)
    X-=X_mean
    y-=y_mean
    t0 = time.time()
    #b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    #b = np.linalg.pinv(X).dot(y) #pseudo inverse
    b = np.linalg.lstsq(X,y,rcond=None)[0]
    e = y_mean - X_mean.dot(b) #y = bx+c
    return T.dot(b) + e

if __name__ == "__main__":
    dataset = load_dataset()
    #dataset = toy_dataset()
    x_train,y_train,x_test,y_test = dataset

    t = time.time()

    #y_pred = nearest_neighbour(x_train,y_train,x_test,k=4)
    #from sklearn.neighbors import KNeighborsRegressor
    #neigh = KNeighborsRegressor(n_neighbors=3)
    #neigh.fit(x_train, y_train)
    #y_pred = neigh.predict(x_test)
    #y_pred = linear_regression(x_train, y_train, x_test)
    forest = random_forest_regressor()
    forest.train(x_train,y_train)
    y_pred = forest.predict(x_test)

    #samples = np.random.choice(x_train.shape[0],600,replace=False)
    #gaussian = gaussian_process_regressor()
    #y_pred = gaussian.train(x_train,y_train,x_test)
    #gaussian.plot()



    print("time",time.time()-t)
    #y_pred = skgp(x_train,y_train,x_test)
    print("MSE",mean_squared_error(y_pred,y_test),y_pred.shape)
    #print(np.abs(y_pred-y_test)[:10])
    visualise(x_test,y_test,y_pred)
