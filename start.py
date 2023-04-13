import pickle
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src import SVM, VertexHistogramKernel, WeisfeilerLehmanKernel


def build_kernels(
    train_data: List[nx.Graph], test_data: List[nx.Graph], kernel: str = "WL"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the kernel matrix for the training and test data.
    :param train_data: The training data
    :param test_data: The test data
    :param kernel: The kernel to use
    :return: The kernel matrix for the training and test data
    """
    if kernel == "WL":
        kernel = WeisfeilerLehmanKernel()
    elif kernel == "VH":
        kernel = VertexHistogramKernel()
    else:
        raise ValueError("Invalid kernel")

    K_train = kernel(train_data, train_data)
    K_test = kernel(test_data, train_data)

    return K_train, K_test


def main(
    K_train: Optional[np.ndarray] = None,
    K_train_test: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    kernel: str = "WL*VH",
) -> np.ndarray:
    if K_train is None or K_train_test is None or train_labels is None:
        train_data = pickle.load(open("data/training_data.pkl", "rb"))
        test_data = pickle.load(open("data/test_data.pkl", "rb"))
        ing_labels = pickle.load(open("data/training_labels.pkl", "rb"))

        if len(kernel.split("*")) > 1:
            K_train, K_train_test = build_kernels(
                train_data, test_data, kernel.split("*")[0]
            )
            K_train_2, K_train_test_2 = build_kernels(
                train_data, test_data, kernel.split("*")[1]
            )
            K_train *= K_train_2
            K_train_test *= K_train_test_2

        else:
            K_train, K_train_test = build_kernels(train_data, test_data, kernel)

    # Train SVM
    svm = SVM()

    if -1 not in train_labels:
        train_labels = 2 * train_labels - 1

    if K_train.shape[0] != K_train_test.shape[0]:
        K_train_test = K_train_test.T

    svm.fit(K_train, train_labels)

    return svm.get_logits(K_train_test)


def format_submission(y_pred, filename="submission.csv") -> None:
    """
    Format the prediction to be submitted to Kaggle.
    :param y_pred: The prediction
    :param filename: The name of the file to save the submission
    :return: None
    """
    Yte = {"Predicted": y_pred}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(filename, index_label="Id")

    return None


if __name__ == "__main__":
    WL_kernel = np.load("data/WL_train.npy")
    WL_kernel_train_test = np.load("data/WL_test_train.npy")

    VH_kernel = np.load("data/K_train_vertex.npy")
    VH_kernel_train_test = np.load("data/K_train_test_vertex.npy")

    train_labels = pickle.load(open("data/training_labels.pkl", "rb"))

    K_train = WL_kernel * VH_kernel
    K_train_test = WL_kernel_train_test * VH_kernel_train_test

    y_pred = main(
        K_train=K_train,
        K_train_test=K_train_test,
        train_labels=train_labels,
        kernel="WL*VH",
    )

    format_submission(y_pred, filename="submission.csv")
