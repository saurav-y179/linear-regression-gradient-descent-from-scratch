This project implements linear regression using gradient descent from scratch, 
without using ML  libraries.

It demonstrates how a model learns parameters (w, b) by minimizing Mean Squared Error (MSE).

WHAT IT DOES:-
- Takes simple (x, y) data
- Initializes parameters (w, b)
- Uses gradient descent to minimize error
- Learns the  least  error line

Maths behind:
- Prediction: ŷ = wx + b
- Loss: Mean Squared Error (MSE)
- Gradients: ∂L/∂w, ∂L/∂b
- Update rule: w = w - lr * dw

FINAL  Result:
The model successfully learns the relationship y = 2x from sample data
Final parameters:
w ≈ 2
b ≈ 0

Visualization by plotting:
data points 
intitial model(pre training)
final model(post training)
jupyter plot:
<img width="801" height="551" alt="Screenshot 2026-04-12 112817" src="https://github.com/user-attachments/assets/b67791c4-1d7a-4eaf-a48e-3882ebe36f2b" />


Notes:
lr effects convergence  speed 
too high:- oscillation
too low:- slow learning 
