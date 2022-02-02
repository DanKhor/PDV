import numpy as np
from numpy.linalg import svd, eig, pinv


def features_mean(X):
    return np.mean(X, axis=0)


def cov_between_class(X, y):
    #matrix B in PDV
    classes, classes_counts = np.unique(y, return_counts=True)
    B = np.zeros((X.shape[1], X.shape[1]))
    m = features_mean(X)
    for i in range(len(classes)):
        X_k = X[np.where(y == classes[i])]
        m_k = features_mean(X_k)
        in_out_diff = m_k - m
        B_k = ((in_out_diff)[:, np.newaxis].dot(in_out_diff[np.newaxis, :])) * classes_counts[i]
        B += B_k
    return B / X.shape[0]


def cov_total(X):
    #matrix T in PDV
    return np.cov(X.T)


def cov_within_class(X, y):
    #matrix W in PDV
    classes, classes_counts = np.unique(y, return_counts=True)
    W = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(classes)):
        X_k = X[np.where(y == classes[i])]
        W_k = np.cov(X_k.T) * classes_counts[i]
        W += W_k
    return W / X.shape[0]


def eigen_find(A, B):
    U, s, _ = svd(B)
    S = np.diag(s)
    S_pinv = pinv(S)
    eig_vals, eig_vec = eig(S_pinv.dot(U.T).dot(A).dot(U).dot(S_pinv))
    #select largest eigenvalues
    eig_pairs = [[np.abs(eig_vals[i]), eig_vec[:, i]] for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return eig_pairs[0][0].real, eig_pairs[0][1].real