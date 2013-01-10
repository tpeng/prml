import numpy as np

A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
x = np.array([[1], [1], [2]])
print A.dot(x)

A = np.array(
    [[4253.96, 6253.96],
     [2919.24, 1919.24],
     [5487.108889, 5487.108889]]
)

print A.argmin(axis=1)