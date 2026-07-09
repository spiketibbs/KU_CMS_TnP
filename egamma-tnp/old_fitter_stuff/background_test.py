import numpy as np
import matplotlib.pyplot as plt

x_min = 50
x_max = 130
x = np.linspace(x_min, x_max, 1000)

def phase_space(x, B, a, b, x_min, x_max):
    shape = (x - x_min)**a * (x_max - x)**b
    shape[(x <= x_min) | (x >= x_max)] = 0
    return B * shape

B = 1
a = 1.2
b = 1.5

y = phase_space(x, B, a, b, x_min, x_max)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Phase Space')
plt.title('Phase Space Function')
plt.grid(True)
plt.show()
