from ._kernel_base import KernelBase
import numpy as np


class TriangularKernel(KernelBase):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x1, x2):
        return max(0, 1 - np.linalg.norm(x1 - x2) / self.sigma)  # type: ignore

    def __repr__(self):
        return f"TriangularKernel(sigma={self.sigma})"

    def __str__(self):
        return f"TriangularKernel(sigma={self.sigma})"

    def __eq__(self, other):
        return self.sigma == other.sigma

    def __hash__(self):
        return hash(self.sigma)
