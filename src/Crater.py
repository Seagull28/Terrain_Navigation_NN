import numpy as np


class Crater:

    def __init__(self, id, centerpoint, diameter, score=0.0):

        self.id = id

        self.centerpoint = np.array(centerpoint)

        self.diameter = diameter

        self.score = score

        self.depth = None

    @property
    def radius(self):

        return self.diameter / 2

    def __str__(self):

        return (
            f"Crater(id={self.id}, "
            f"center={self.centerpoint}, "
            f"diameter={self.diameter}, "
            f"depth={self.depth})"
        )

    def __repr__(self):

        return self.__str__()