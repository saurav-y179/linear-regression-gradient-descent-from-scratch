#gradient  descent python 
import matplotlib.pyplot   as  plt 
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

y_before = [0 * xi + 0 for xi in x]
def compute_mse(x:list[float],y:list[float],w:float,b:float) -> float:
    """Compute Mean sqaured error  between   true  and  predicted  values 
    Args:
        x :input  features 
        y:True labels
        w: Weight
        b:bias 

    Returns:
        MSE value
    """
    n = len(y)
    total = 0
    for i in range(n):
        error = y[i]  - (w*x[i] +b )
        total += error**2
    return total/n
def compute_gradients (x:list[float],y:list[float],w:float,b:float)  ->tuple[float,float]:
    """Compute gradients  of MSE  w.r.t w  and  b 
        x :input  features 
        y:True labels
        w: Weight
        b:bias 

    Returns:
        Tuple of(dw,db)
    """
    n = len(y)
    dw,db = 0.0,0.0
    for i in range(n):
        error = y[i] - (w*x[i]+b)
        dw += ( -2/n)* error *x[i]
        db += (-2/n)* error 
    return dw,db 
def train(x: list[float],y: list[float],lr: float = 0.1, epochs : int = 1000) ->tuple[float,float,list[float]]:
    """Trains linear regression using gradient descent .
        x :input  features 
        y:True labels
        lr

    Returns:
        Tuple of(final w ,final b , MSE history)
    """
    w,b = 0.0,0.0
    MSE_history = []
    for k in range(epochs+1 ):
        dw,db = compute_gradients(x,y,w,b)
        w -= lr*dw
        b -= lr*db 
        MSE_history.append(compute_mse(x,y,w,b))
    return w,b,MSE_history
def plot_results(x: list[float], y: list[float], y_before: list[float], y_after: list[float], w: float, b: float) -> None:
    """
    Plots true data, predictions before and after training.

    Args:
        x: Input features
        y: True labels
        y_before: Predictions before training
        y_after: Predictions after training
        w: Final weight
        b: Final bias
    """
    plt.scatter(x, y, label="data")
    plt.plot(x, y_before, label="Before training (w=0, b=0)")
    plt.plot(x, y_after, label=f"After training (w={w:.2f}, b={b:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Learning")
    plt.legend()
    plt.show()

w, b, MSE_history = train(x, y)
y_after = [w * xi + b for xi in x]

print(f"w: {w:.4f} | b: {b:.4f} | MSE: {MSE_history[-1]:.6f}")
plot_results(x, y, y_before, y_after, w, b)