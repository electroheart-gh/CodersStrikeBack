import numpy as np
import math
import sys
# import matplotlib.pyplot as plt


class DebugTool:
    def __init__(self):
        try:
            self.fd = open(r"C:\Users\JUNJI\Documents\Condingame\pyCharmProject\CodersStrikeBack\input.txt")
            import matplotlib.pyplot as plt
            self.plt = plt
            self.debug_mode = True
        except (ImportError, OSError):
            self.debug_mode = False

    def input(self):
        if self.debug_mode:
            data = self.fd.readline()
        else:
            data = input()
        print(data, file=sys.stderr)
        return data

    @staticmethod
    def stderr(*args):
        print(*args, file=sys.stderr)

    def plot_vector_clock(self, vct, clr="b", txt=""):
        self.plt.plot((0, vct[0]), (0, vct[1]), color=clr)
        self.plt.text(vct[0], vct[1], txt)


class Vector(np.ndarray):
    def __new__(cls, x, y):
        vctr = np.r_[x, y]
        return vctr.view(cls)

    def magnitude(self):
        return np.linalg.norm(self)

    def x(self):
        return self[0]

    def y(self):
        return self[1]

    def normalized(self):
        if self.magnitude() == 0:
            return Vector(0, 0)
        else:
            copy = self.copy()
            copy = copy / copy.magnitude()
            return copy

    def as_magnitude(self, scalar):
        return self.normalized() * scalar

    def abs_angle(self):
        """Returns radians between 0 and 2 * pi for absolute angle.
        In THIS GAME field, 0 means facing EAST while 90 means facing SOUTH."""
        # Flip because atan2() takes y then x
        angle = math.atan2(*np.flipud(self))

        # Make angle hold radians between 0 and 2 * pi
        # because atan2() returns radians between -pi and pi
        if angle < 0:
            angle += math.pi * 2
        return angle

    def angle_for(self, vector):
        """Returns radians between -pi and pi.
          In THIS GAME field, POSITIVE number means CLOCKWISE direction from SELF to VECTOR."""
        diff = vector.abs_angle() - self.abs_angle()
        # print("vctr", vector.abs_angle(), "self", self.abs_angle(), file=sys.stderr)
        if abs(diff) < math.pi:
            return diff
        elif self.abs_angle() < math.pi:
            return diff - math.pi * 2
        else:
            return diff + math.pi * 2

    def distance_from(self, point_as_vector):
        """Returns the minimum distance from the point(vector) to the line segment(self)."""
        if np.dot(self, point_as_vector) < 0:
            return point_as_vector.magnitude()
        elif np.dot(-self, point_as_vector - self) < 0:
            return (point_as_vector - self).magnitude()
        else:
            print(self, point_as_vector, np.cross(self, point_as_vector), self.magnitude(), file=sys.stderr)
            return abs(np.cross(self, point_as_vector) / self.magnitude())

    def rotate(self, r):
        rot = np.matrix(((math.cos(r), math.sin(r)), (-math.sin(r), math.cos(r))))
        # print(np.dot(self, rot), file=sys.stderr)
        # print(np.array(np.dot(self, rot)).ravel(), file=sys.stderr)
        return Vector(*np.array(np.dot(self, rot)).ravel())


DT = DebugTool()

DT.plt.figure(1)
DT.plt.plot((0, 99.7), (0, 8.1), c="b")  # straight to CP
DT.plt.plot((0, -8.1), (0, 99.7), c="y")  # rotate for compo of inertia (rot -1.57)

DT.plt.figure(2)
DT.plt.plot((0, 99.7), (0, 7.5), c="b")  # straight to CP
DT.plt.plot((0, 7.5), (0, -99.7), c="y")  # rotate for compo of inertia
DT.plt.plot((0, 99.6), (0, 8.7), c="r")  # rotate for compo and abs angle
DT.plt.plot((0, 92.0), (0, 39.0), c="black")  # original angle as vector
DT.plt.plot((0, 545.0), (0, 7.0), c="y")  # inertia

DT.plt.ylim(150, -150)
DT.plt.xlim(-150, 150)
DT.plt.show()

if DT.debug_mode:
    DT.plt.figure("test vector clock")
    tv = Vector(-1, -1)
    DT.plot_vector_clock(tv, "black", "Original")
    DT.plot_vector_clock(tv.rotate(math.pi * 0.2), "b", "PI*0.2")
    DT.plot_vector_clock(tv.rotate(math.pi * 0.5), "b", "PI*0.5")
    DT.plot_vector_clock(tv.rotate(math.pi * 0.7), "b", "PI*0.7")
    DT.plot_vector_clock(tv.rotate(math.pi * 1.0), "r", "PI")
    DT.plot_vector_clock(tv.rotate(math.pi * 1.4), "r", "PI*1.4")
    DT.plot_vector_clock(tv.rotate(math.pi * 1.5), "y", "PI*1.5")
    DT.plot_vector_clock(tv.rotate(math.pi * 1.8), "y", "PI*1.8")
    DT.plot_vector_clock(tv.rotate(math.pi * 2), "g", "PI*2")
    DT.plot_vector_clock(tv.rotate(math.pi * 2.3), "g", "PI*2.3")
    DT.plt.xlim(-2, 2)
    DT.plt.ylim(2, -2)
    pass





# class Vector2d(np.ndarray):
#     def magnitude(self):
#         return np.linalg.norm(self)
#
#     def __str__(self):
#         pass
#
#     def __repr__(self):
#         pass
#
#
# class Vector2d(np.ndarray):
#     def __new__(cls, x, y):
#         return np.array([x, y])
