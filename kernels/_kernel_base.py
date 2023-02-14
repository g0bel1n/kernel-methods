from abc import ABC, abstractmethod
import numpy as np

class KernelBase(ABC):
    @abstractmethod
    def __call__(self, x1, x2):
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

    def dist_matrix(self, x1, x2):
        dist_mat = np.array([[self(x1[i], x2[j]) for j in range(len(x2))] for i in range(len(x1))])
        dist_mat += np.eye(len(x1)) * np.inf
        return dist_mat

    def gram_matrix(self, x):
        return self.dist_matrix(x, x)

    

    
