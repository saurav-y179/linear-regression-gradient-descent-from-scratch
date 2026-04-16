# Linear Regression with Gradient Descent (From Scratch)

A clean, minimal implementation of **univariate linear regression** trained with **gradient descent**, written in pure Python.

This project is intended for learning and interview-style demonstration. It avoids machine learning frameworks so the optimization process is explicit and easy to follow.

## Overview
Given a dataset:
{(x_i, y_i)} for i = 1 to n

Model:
ŷ_i = w x_i + b

Loss function (Mean Squared Error):
L(w, b) = (1/n) * Σ_{i=1}^{n} (y_i - ŷ_i)^2

Substituting ŷ_i:
L(w, b) = (1/n) * Σ_{i=1}^{n} (y_i - (w x_i + b))^2

Gradients:
∂L/∂w = (-2/n) * Σ_{i=1}^{n} x_i (y_i - (w x_i + b))

∂L/∂b = (-2/n) * Σ_{i=1}^{n} (y_i - (w x_i + b))

Update rules (Gradient Descent):
w ← w - α * (∂L/∂w)

b ← b - α * (∂L/∂b)

where α > 0 is the learning rate

## Features

- Pure Python implementation of linear regression
- Explicit gradient calculation (no autograd)
- Training loop with configurable learning rate and epochs
- MSE history tracking
- Visualization of:
  - raw data
  - pre-training predictions
  - post-training fitted line

## Project Structure

- `lin-regression  gradient descent python.py` — main script containing training and plotting logic

## Example

Sample data in the script follows the linear relationship:

- `x = [1, 2, 3, 4]`
- `y = [2, 4, 6, 8]`

Expected outcome after training:

- `w ≈ 2`
- `b ≈ 0`

## How to Run

1. Install dependencies:

   ```bash
   pip install matplotlib
   ```

2. Run the script:

   ```bash
   python "lin-regression  gradient descent python.py"
   ```

The program prints final parameter values and displays a plot of the learned line.

## Notes on Learning Rate

- Too high: training may oscillate or diverge
- Too low: training converges slowly

A moderate value (for example, `0.1` in this dataset) works well.
