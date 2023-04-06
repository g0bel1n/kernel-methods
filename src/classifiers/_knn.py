from ._classifier_base import ClassifierBase
from kernels._base import KernelBase
import numpy as np


class KNNClassifier(ClassifierBase):
    def __init__(self, k: int, kernel: KernelBase):
        super().__init__()
        self.k = k
        self.kernel = kernel

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.set_classes(np.unique(y))
        self.dist_matrix = self.kernel.gram_matrix(X)

    def predict(self, X):
        y = np.zeros(len(X))
        idxs = np.argsort(self.dist_matrix, axis=1)[: self.k]
        print(idxs.shape)
        y = np.bincount(self.y[idxs], minlength=len(self.get_classes()))  # type: ignore
        return y

    def __repr__(self):
        return f"KNNClassifier(k={self.k}, kernel={self.kernel})"

    def __str__(self):
        return f"KNNClassifier(k={self.k}, kernel={self.kernel})"

    def __eq__(self, other):
        return self.k == other.k and self.kernel == other.kernel

    def __hash__(self):
        return hash((self.k, self.kernel))
