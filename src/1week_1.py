import numpy as np
import matplotlib.pyplot as plt

theta = np.deg2rad(45) 
v0 = 100.0           
g = 9.81            
h = 100.0            


a = 0.5 * g
b = -v0 * np.sin(theta)
c = -h
t_flight = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


t = np.linspace(0, t_flight, num=100)



x = v0 * np.cos(theta) * t
y = h + v0 * np.sin(theta) * t - 0.5 * g * t**2

x_zero = x[-1]      
y_max = np.max(y)
x_max = x[np.argmax(y)]  

print("최대 x 길이:", x_zero)
print("최고 y 높이:", y_max)
print("최고일대 x : ", x_max)


plt.plot(x, y, label='parabola')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()
