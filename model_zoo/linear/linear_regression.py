import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class LinearRegression:
    """
    Simple implementation of unregularized linear regression using gradient descent.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        """
        Initialize model.

        Args:
            lr (float): Learning rate for gradient descent. Defaults to 0.01.
            epochs (int): Number of training iterations. Defaults to 1000.
        """
        self.N = None
        self.p = None
        self.weights = None
        self.bias = None

        self.lr = lr
        self.epochs = epochs

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """
        Train the model using gradient descent.

        Args:
            X (np.ndarray): Input feature matrix of shape (N, p).
            y (np.ndarray): Target vector of shape (N,).
        """
        self.N, self.p = X.shape  # N instances, p features

        self.weights = np.zeros(self.p)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            # calculate gradient
            dw = (2.0 / self.N) * np.dot(X.T, y_pred - y)
            db = (2.0 / self.N) * np.sum(y_pred - y)

            # update
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray:
        """
        Predict target values using the learned linear model.

        Args:
            X (np.ndarray): Input feature matrix of shape (N, p).

        Returns:
            np.ndarray: Predicted target values of shape (N,).
        """
        return np.dot(X, self.weights) + self.bias
