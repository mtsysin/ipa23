import torch
import matplotlib.pyplot as plt
import math

epochs = 100

lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine

x = [i for i in range(epochs)]
y = [lf(i) for i in range(epochs)]

plt.plot(x, y)
plt.savefig("./out/fun.png")
