import numpy as np
from cvxopt import matrix, solvers

from ._classifier_base import ClassifierBase


def svm_solver(K, y, C=1.0):
    n_samples = K.shape[0]
    P = np.outer(y, y) * K
    q = -np.ones(n_samples)
    G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
    h = np.hstack((np.zeros(n_samples), C * np.ones(n_samples)))
    A = y[np.newaxis, :].astype("float64")
    b = np.zeros(1)

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol["x"]).flatten()


def train(K, y, C=1.0):
    alphas = svm_solver(K, y, C)
    support_vectors = alphas > 1e-6
    ind = np.arange(len(alphas))[support_vectors]
    alphas = alphas[support_vectors]
    support_y = y[support_vectors]
    support_K = K[ind][:, support_vectors]

    b = np.mean(support_y - np.sum(support_K * alphas * support_y, axis=1))

    return alphas, support_y, ind, b


def get_logits(K_test, alphas, support_y, ind, b):
    return np.dot(K_test[ind].T, alphas * support_y) + b


class SVM(ClassifierBase):
    def __init__(self, C=1.0):
        super().__init__()
        self.C = C
        self.alphas = None
        self.support_y = None
        self.ind = None
        self.b = None

    def fit(self, K, y):
        self.alphas, self.support_y, self.ind, self.b = train(K, y, self.C)

    def get_logits(self, K_test):
        return get_logits(K_test, self.alphas, self.support_y, self.ind, self.b)
