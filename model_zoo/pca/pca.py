import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class PCA:
    """
    Principal Component Analysis (PCA) implementation via eigendecomposition of covariance matrix.
    """

    def __init__(self, n_components: int):
        """
        Initialize the PCA model.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.comps = None
        self.N = None
        self.p = None

    def fit(self, X: npt.NDArray[np.float64]):
        """
        Fit the PCA model to the dataset by computing the top principal components.

        Args:
            X (np.ndarray): Data matrix of shape (N, p), where N is the number of samples
                and p is the number of features.
        """
        self.N, self.p = X.shape

        # zero-mean features
        self.mean = np.mean(X, axis=0)
        print(self.mean.shape)
        X = X - self.mean

        # find eigenvalues and eigenvectors of covariance matrix
        cov = np.cov(X.T)
        eigenvals, eigenvecs = np.linalg.eig(cov)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]

        # take only the top N components
        self.comps = eigenvecs[:, : self.n_components]  # [p, n_components]

    def project(self, X: npt.NDArray[np.float64]):
        """
        Project new data onto the principal components learned during fit.

        Args:
            X (np.ndarray): New data matrix of shape (N, p) to be projected.

        Returns:
            np.ndarray: Projected data matrix of shape (N, n_components).
        """
        assert self.p == X.shape[1], "Input feature dim must match that of training."
        X = X - self.mean
        return X @ self.comps
