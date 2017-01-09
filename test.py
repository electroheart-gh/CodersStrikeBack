import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot((0, 99.7), (0, 8.1), c="b")        # straight to CP
plt.plot((0, -8.1), (0, 99.7), c="y")        # rotate for compo of inertia (rot -1.57)

plt.figure(2)
plt.plot((0, 99.7), (0, 7.5), c="b")        # straight to CP
plt.plot((0, 7.5), (0, -99.7), c="y")       # rotate for compo of inertia
plt.plot((0, 99.6), (0, 8.7), c="r")        # rotate for compo and abs angle
plt.plot((0, 92.0), (0, 39.0), c="black")   # original angle as vector
plt.plot((0, 545.0), (0, 7.0), c="y")      # inertia

plt.ylim(150, -150)
plt.xlim(-150, 150)
plt.show()


class Vector2d(np.ndarray):
    def magnitude(self):
        return np.linalg.norm(self)

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Vector2d(np.ndarray):
    def __new__(cls, x, y):
        return np.array([x, y])
