import numpy as np

class LinearRegressor:

    def __init__(self) -> None:
        self.w = None
        
    def X_bias(self, X):
        return np.hstack((np.ones((len(X),1)), X))

    def predict(self, X):
        X = self.X_bias(X)
        return X.dot(self.w)


class OrdinaryLeastSquares(LinearRegressor):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        X = self.X_bias(X)

        # OLS weight function:
        # w = (X^T * X)^-1 * X^T * y
        self.w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)


class RidgeRegressor(LinearRegressor):

    def __init__(self, l2_penalty=1.0, learnig_rate=0.1, iterations=1000) -> None:
        super().__init__()
        self.l2_penalty = l2_penalty
        self.ds = learnig_rate
        self.it = iterations

    def fit(self, X, y):
        self.X_train = X
        self.X = self.X_bias(X)
        self.y = y
        self.n, self.p = self.X.shape #n:observation, p:features+1

        # w = (X^T * X + s*I)^-1 * X^T * y
        self.w = np.linalg.inv(self.X.T.dot(self.X) + self.l2_penalty*np.identity(self.p)).dot(self.X.T).dot(self.y)
        for i in range(self.it):
            self.update_w()

    def update_w(self):
        # Ridge cost function:
        # f = |y - y'|^2_2 - s*|w|^2_2
        dw = np.zeros(self.p)
        dy = self.y - self.predict(self.X_train)

        for i in range(self.p):
            dw[i] = 2 * (self.l2_penalty*self.w[i] - self.X[:,i].dot(dy)) / self.n

        self.w = self.w - self.ds*dw


class Lasso(LinearRegressor):

    def __init__(self, l1_penalty=1.0, learnig_rate=0.1, iterations=1000) -> None:
        super().__init__()
        self.l1_penalty = l1_penalty
        self.ds = learnig_rate
        self.lit = iterations

    def fit(self, X, y):
        self.X_train = X
        self.X = self.X_bias(X)
        self.y = y
        self.n, self.p = self.X.shape #n:observation, p:features+1

        # w = (X^T * X + s*I)^-1 * X^T * y
        self.w = np.linalg.inv(self.X.T.dot(self.X) + self.l1_penalty*np.identity(self.p)).dot(self.X.T).dot(self.y)
        for i in range(self.lit):
            self.update_w()

    def update_w(self):
        # Lsso cost function:
        # f = |y - y'|^2_2 + s*|w|
        dw = np.zeros(self.p)
        dy = self.y - self.predict(self.X_train)

        for i in range(self.p):
            if self.w[i] >= 0:
                dw[i] = -2 * self.X[:,i].dot(dy) + self.l1_penalty
            else:
                dw[i] = -2 * self.X[:,i].dot(dy) - self.l1_penalty
        dw /= self.n

        self.w = self.w - self.ds*dw