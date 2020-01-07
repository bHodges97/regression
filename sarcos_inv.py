import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import time

from regression import *
from randomforest import *
from gaussian import *

from utils import *

from sklearn.ensemble import RandomForestRegressor

def toy_dataset(size=200):
    np.random.seed(0)
    noise = np.random.normal(0,0.2,(size,1))
    x = np.random.uniform(0, 5,(size,1))
    x = np.sort(x)
    y = np.sin(x)#.ravel()
    #y = (x * 1.5 ) + 5
    #y += noise
    #y = 1/5*np.sin(x*np.pi*2*5) + 2*x
    return np.hstack([x,y])

def load_dataset():
    data = np.genfromtxt('sarcos_inv.csv', delimiter=',')
    np.random.shuffle(data)
    return data

def split_data(data, ratio = 0.8):
    rows = int(data.shape[0] * ratio)
    x_train = data[:rows,:-1]
    y_train = data[:rows,-1].ravel()
    x_test = data[rows:,:-1]
    y_test = data[rows:,-1].ravel()
    return (x_train,y_train,x_test,y_test)

def visualise(x_test,y_test,y_pred):
    plt.scatter(x_test, y_test, color='darkorange', label='data' ,s=5)
    plt.scatter(x_test, y_pred, color='navy', label='prediction', s=1)
    plt.show()

if __name__ == "__main__":
    dataset = load_dataset()
    #dataset = toy_dataset()

    x_train,y_train,x_test,y_test = split_data(dataset)
    t = time.time()

    #regressor = GaussianProcessRegressor()
    #regressor = RandomForestRegressor()
    #----
    #regressor = knn_regressor()
    #regressor = linear_regressor()
    regressor = gaussian_process_regressor()
    #regressor = random_forest_regressor()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)

    #samples = np.random.choice(x_train.shape[0],600,replace=False)
    #y_pred = gaussian.train(x_train,y_train,x_test)
    #gaussian.plot()



    print("time",time.time()-t)
    #y_pred = skgp(x_train,y_train,x_test)
    print("MSE",mean_squared_error(y_pred,y_test))
    #print(np.abs(y_pred-y_test)[:10])
    visualise(x_test,y_test,y_pred)
