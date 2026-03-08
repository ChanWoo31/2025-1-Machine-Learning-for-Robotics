import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
    
def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad
    
def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad
    
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

x = np.array([2.0, 3.0])
a = gradient_descent(function_2, x, lr = 0.01, step_num = 100)
print(a)

