import numpy as np

class linear_regressor:
    def __init__(self, method = "pinv"):
        if method == "pinv": #pseudo inverse
            self.solver = lambda X,y: np.linalg.pinv(X).dot(y)
        elif method == "inv":
            self.solver = lambda X,y: np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        else:
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

