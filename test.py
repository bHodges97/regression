import numpy as np


def rbf(x1,x2,r2 = 1):
    print("t",(x1[...,None]-x2.T[None,...]).shape)
    pw = np.sum(x1[...,None]-x2.T[None,...],axis=1) ** 2
    return np.exp( - pw / 1 )

def r1(x1,x2):
    cov = lambda x1,x2: np.exp((-np.sum((x1-x2)**2)) / 1)
    return np.reshape([cov(x,y) for x in x1 for y in x2], (x1.shape[0], x2.shape[0]))

def r2(x1,x2):
    pw = np.sum(np.power(x1[:,None]-x2[None,:],2),axis=2)
    return np.exp( - pw / 1 )

def r3(x1,x2):
    dist = x1[:,None]-x2[None,:]
    norm = np.einsum('ijk,ijk->ij',dist,dist)
    return np.exp( -norm  / 1)

def r4(x1,x2):
    #norm = np.einsum('ij,ij->i',x1,x1)[:,None] + np.einsum('ij,ij->i',x2,x2) - 2 * np.dot(x1, x2.T)
    norm = np.einsum('ij,ij,kl,kl->ki',x1,x1,x2,x2) - 2 * np.dot(x1, x2.T)
    return np.exp( -norm  / 1)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
x = np.arange(2000).reshape(400,5)
#x = np.arange(10).reshape(5,2)

import time

print(np.allclose(r3(x,x),r4(x,x)))
t = time.time()
for i in range(1000):
    r4(x,x)
print("time",time.time()-t)

print(x.shape)

