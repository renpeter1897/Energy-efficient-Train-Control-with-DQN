import numpy as np
a = []
for _ in range(296):
    a.append(2)
for i in range(2, len(a)+1):
    a[-i] += 0.99 * a[-i+1]