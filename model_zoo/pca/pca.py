import numpy as np
import numpy.typing as npt

from model_zoo.utils import setup_logger

logger = setup_logger()


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.comps = None
        self.N = None
        self.p = None

    def fit(self, X: npt.NDArray[np.float64]):
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
        assert self.N, self.p == X.shape
        X = X - self.mean
        return X @ self.comps
