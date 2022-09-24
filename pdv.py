import numpy as np
import time
from PDV.matrix_computations import cov_between_class, cov_total, eigen_find, cov_within_class


class PDV:
    def __init__(
            self,
            n_components=None,
            lambda_=0,
            random_seed=42
    ):
        """
        Parameters
        ----------
        n_components: int, decomposition components
        alpha: float, between 0 and 1 balanced between PCA and FLDA
        random_seed: int, state of np.random.seed

        """
        self.n_components = n_components
        self.lambda_ = lambda_
        self.random_seed = random_seed

        self.components = None

        self.explained_variance_ratio_ = None

    def fit(self, X, y):
        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])
        B = cov_between_class(X, y)
        T = cov_total(X)
 #       T = cov_within_class(X, y)
        P = np.eye(B.shape[0])

        #создадим массив компонент
        components = np.zeros((self.n_components, B.shape[0]))

        for i in range(self.n_components):
            A_ = self.lambda_*(((P.T).dot(B)).dot(P)) + (1 - self.lambda_)*(((P.T).dot(T)).dot(P))
            B_ = self.lambda_*(((P.T).dot(T)).dot(P)) + (1 - self.lambda_)*np.eye(B.shape[0])

            alpha, a = eigen_find(A_, B_)
            components[i] = a
            P = (np.eye(B.shape[0]) - a[:, np.newaxis].dot(a[np.newaxis, :])).dot(P)

        self.components = components
        self.explained_variance_ratio_ = np.diag(np.cov(X.dot(self.components.T).T)) / np.sum(np.diag(T))

        return self

    def transform(self, X):
        if self.components is not None:
            return X.dot(self.components.T)
        else:
            raise ValueError('PDV components is empty')

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


