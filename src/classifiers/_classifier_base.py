from abc import ABC, abstractmethod
import numpy as np


class ClassifierBase(ABC):

    def __init__(self):
        self._classes = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_classes(self):
        return self._classes

    def set_classes(self, classes):
        self._classes = classes