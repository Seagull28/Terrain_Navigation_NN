import numpy as np

def extractCenterpoints(craters):
    return {k: c.centerpoint for k, c in craters.items()}