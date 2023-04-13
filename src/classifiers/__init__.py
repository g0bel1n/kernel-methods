import cvxopt

from ._svm import SVM

__all__ = ["SVM"]

cvxopt.solvers.options["show_progress"] = False
