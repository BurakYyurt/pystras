import numpy as np


def engineering_stress(strain, C):
    return np.dot(C, strain)
