import numpy as np
import matplotlib.pyplot as plt


init_v = 10
angle = np.deg2rad(45)
g = 9.81

h = 100

t = np.arange(0, (init_v * np.sin(angle) + np.sqrt((init_v * np.sin(angle))**2 + 2*g*h)) / g, 0.1)


x = init_v * np.cos(angle) * t
y = h + init_v * np.sin(angle) *t -0.5 * g * t **2

plt.plot(x, y)
plt.show()