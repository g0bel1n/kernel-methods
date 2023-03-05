from ._classifier_base import ClassifierBase
from kernels._kernel_base import KernelBase

class SVMClassifier(ClassifierBase):

    def __init__(self, kernel: KernelBase, C: float, tol: float, max_iter: int):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def __repr__(self):
        return f"SVMClassifier(kernel={self.kernel}, C={self.C}, tol={self.tol}, max_iter={self.max_iter})"

    def __str__(self):
        return f"SVMClassifier(kernel={self.kernel}, C={self.C}, tol={self.tol}, max_iter={self.max_iter})"
    