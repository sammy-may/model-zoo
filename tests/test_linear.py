import numpy as np

from model_zoo.linear.linear_regression import LinearRegression


def test_linear_regression():
    X = np.array([[1, 1], [1, 2], [2, 1], [-1, 0]])
    y = np.array([2, 1, 4, -1])

    lr = LinearRegression(lr=0.1, epochs=1000)
    lr.fit(X, y)

    assert np.allclose(lr.weights, np.array([2, -1]), rtol=1e-6)

    assert np.allclose(y, lr.predict(X), rtol=1e-6)
