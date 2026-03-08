import numpy as np

# 임의의 x0, x1 값에 대하여 f의 최소값을 gradient descent를 이용하여 구하기
x0, x1 = 3.0, 4.0

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx] 
        
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x) 
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)  
        
        grad[idx] = (fxh1 - fxh2) / (2*h) 
        
        x[idx] = tmp_val # 값 복원
        
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        grad = _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X) 
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
            
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
    
a = gradient_descent(function_2, np.array([x0, x1]), lr = 0.1, step_num = 100)
print(a)
