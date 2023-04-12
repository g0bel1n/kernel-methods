from abc import ABC, abstractmethod
import numpy as np


class ClassifierBase(ABC):
    

    @abstractmethod
    def fit(self, X, y):
        ...

    def predict(self, X):
        return self.get_logits(X).argmax(axis=1)
        

    @abstractmethod
    def get_logits(self, K) -> np.ndarray:
        ...

