import numpy as np
from numpy.linalg import inv

class PLS_DA:

    def __init__(self):
        """
        Parameters
        ----------

        B: np.array, regression coefficient vector
        """
        self.B = None

    def fit(self, X, y, p):
        """
        Parameters
        ----------

        X: np.array, data matrix
        y: np.array, response vector
        p: int, fixed number of steps in the algorithm
        """
        E_0 = X.copy()
        F_0 = y.copy()

        for i in range(p):
            W_p = (E_0.T).dot(F_0)
            temp_matrix = np.power((((W_p.T).dot(E_0.T)).dot(E_0)).dot(W_p), -0.5)
            T_p = (E_0.dot(W_p)).dot(temp_matrix)
            P_p = (E_0.T).dot(T_p)
            Q_p = (F_0.T).dot(T_p)
            E_0 = E_0 - T_p.dot(P_p.T)
            F_0 = F_0 - T_p.dot(Q_p.T)

        self.W = W_p
        self.T = T_p
        self.P = P_p
        self.Q = Q_p
        self.B = (W_p.dot(inv((P_p.T).dot(W_p)))).dot(Q_p.T)

        return self

    def predict(self, X):
        return X.dot(self.B)

    def fit_predict(self, X, y, p):
        self = self.fit(X, y, p)
        return self.predict(X)