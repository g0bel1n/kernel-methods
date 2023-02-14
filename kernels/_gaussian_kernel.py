from ._kernel_base import KernelBase   
import numpy as np


class GaussianKernel(KernelBase):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))

    def __repr__(self):
        return f"GaussianKernel(sigma={self.sigma})"

    def __str__(self):
        return f"GaussianKernel(sigma={self.sigma})"

    def __eq__(self, other):
        return self.sigma == other.sigma

    def __hash__(self):
        return hash(self.sigma)

    