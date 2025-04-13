import numpy as np
import numpy.typing as npt

from model_zoo.utils import setup_logger

logger = setup_logger()


class LinearRegression:
    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        self.N = None
        self.p = None
        self.weights = None
        self.bias = None

        self.lr = lr
        self.epochs = epochs

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
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
        return np.dot(X, self.weights) + self.bias
