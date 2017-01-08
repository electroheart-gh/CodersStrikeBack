import numpy


#
class Vector2d(numpy.ndarray):
    def magnitude(self):
        return numpy.linalg.norm(self)

    def __str__(self):
        pass

    def __repr__(self):
        pass




class Vector2d(numpy.ndarray):
    def __new__(cls, x, y):
        return numpy.array([x, y])
