from kernels import GaussianKernel

from classifiers import KNNClassifier

kernel = GaussianKernel(1)
knn = KNNClassifier(3, kernel)
