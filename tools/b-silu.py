import numpy as np
import matplotlib.pyplot as plt

def b_silu(x, alpha=1.67):
    return (x + alpha) * (1 / (1 + np.exp(-x))) - alpha / 2

x = np.linspace(-6, 6, 400)
y = b_silu(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y, label='B-SiLU (α=1.67)')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('B-SiLU Activation Function')
plt.xlabel('x')
plt.ylabel('B-SiLU(x)')
plt.legend()
plt.grid(True)
plt.show()