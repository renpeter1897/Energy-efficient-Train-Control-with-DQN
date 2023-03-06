import matplotlib
import matplotlib.pyplot as plt


def ReLU(x):
    return max(x, 0)

a, b = [], []
for i in range(-3, 4):
    a.append(i)
    b.append(ReLU(i))

plt.plot(a, b)
plt.title('ReLU')
plt.savefig('./ReLU.svg', format='svg')
plt.show()