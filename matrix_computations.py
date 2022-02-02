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
        in_out_diff = np.mean(X[np.where(y == classes[i])], axis=0) - m
        B += (in_out_diff)[:, np.newaxis].dot(in_out_diff[np.newaxis, :]) / classes_counts[i]
    return B / X.shape[0]


def cov_total(X):
    #matrix T in PDV
    T = np.zeros((X.shape[1], X.shape[1]))
    m = features_mean(X)
    for i in range(X.shape[0]):
        obj_centr = X[i] - m
        T += obj_centr[:, np.newaxis].dot(obj_centr[np.newaxis, :])
    return T / X.shape[0]


def cov_within_class(X, y):
    #matrix W in PDV
    classes, classes_counts = np.unique(y, return_counts=True)
    W = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(classes)):
        X_k = X[np.where(y == classes[i])]
        m_k = features_mean(X_k)
        W += ((X_k - m_k).T).dot((X_k - m_k))
    return W / X.shape[0]


def eigen_find(A, B):
    U, s, _ = svd(B)
    S = np.diag(s)
    S_pinv = pinv(S)
    eig_vals, eig_vec = eig(S_pinv.dot(U.T).dot(A).dot(U).dot(S_pinv))

    return np.real(eig_vals[0]), np.real(eig_vec[:, 0])