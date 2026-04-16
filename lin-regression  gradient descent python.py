"""Linear regression with gradient descent (from scratch).

This module demonstrates how to train a univariate linear regression model
without machine learning libraries.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


X_DATA = [1.0, 2.0, 3.0, 4.0]
Y_DATA = [2.0, 4.0, 6.0, 8.0]


def compute_mse(x: list[float], y: list[float], w: float, b: float) -> float:
    """Compute mean squared error between true and predicted values."""
    n = len(y)
    total = 0.0
    for i in range(n):
        error = y[i] - (w * x[i] + b)
        total += error**2
    return total / n


def compute_gradients(
    x: list[float], y: list[float], w: float, b: float
) -> tuple[float, float]:
    """Compute gradients of MSE with respect to weight and bias."""
    n = len(y)
    dw, db = 0.0, 0.0

    for i in range(n):
        error = y[i] - (w * x[i] + b)
        dw += (-2 / n) * error * x[i]
        db += (-2 / n) * error

    return dw, db


def train(
    x: list[float], y: list[float], lr: float = 0.1, epochs: int = 1000
) -> tuple[float, float, list[float]]:
    """Train linear regression parameters using gradient descent."""
    w, b = 0.0, 0.0
    mse_history: list[float] = []

    for _ in range(epochs):
        dw, db = compute_gradients(x, y, w, b)
        w -= lr * dw
        b -= lr * db
        mse_history.append(compute_mse(x, y, w, b))

    return w, b, mse_history


def plot_results(
    x: list[float],
    y: list[float],
    y_before: list[float],
    y_after: list[float],
    w: float,
    b: float,
) -> None:
    """Plot data and model predictions before and after training."""
    plt.scatter(x, y, label="Data points")
    plt.plot(x, y_before, label="Before training (w=0, b=0)")
    plt.plot(x, y_after, label=f"After training (w={w:.2f}, b={b:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression via Gradient Descent")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run training and visualize results."""
    y_before = [0.0 for _ in X_DATA]

    w, b, mse_history = train(X_DATA, Y_DATA)
    y_after = [w * xi + b for xi in X_DATA]

    print(f"Final parameters -> w: {w:.4f}, b: {b:.4f}, MSE: {mse_history[-1]:.6f}")
    plot_results(X_DATA, Y_DATA, y_before, y_after, w, b)


if __name__ == "__main__":
    main()
