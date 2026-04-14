import numpy as np

class Crater:
    def __init__(self, id, centerpoint, diameter, score=0.0):
        self.id = id
        self.centerpoint = np.array(centerpoint)  # 🔥 FIX
        self.diameter = diameter
        self.score = score

        # 🔥 Always initialize depth
        self.depth = None

    def __str__(self):
        return f"Crater(id={self.id}, center={self.centerpoint}, diameter={self.diameter}, depth={self.depth})"

    def __repr__(self):
        return self.__str__()