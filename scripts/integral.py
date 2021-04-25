import numpy as np


def gauss_3D(function, scheme=3):
    weights = [[2],
               [1, 1],
               [5 / 9, 8 / 9, 5 / 9]]
    points = [[0],
              [-1 / np.sqrt(3), 1 / np.sqrt(3)],
              [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]]

    gauss_wgh = weights[scheme - 1]
    gauss_loc = points[scheme - 1]

    result = 0
    for xi, wx in zip(gauss_loc, gauss_wgh):
        for eta, wy in zip(gauss_loc, gauss_wgh):
            for zeta, wz in zip(gauss_loc, gauss_wgh):
                result += function(xi, eta, zeta) * wx * wy * wz

    return result


def gauss_2D(function, scheme=3):
    weights = [[2],
               [1, 1],
               [5 / 9, 8 / 9, 5 / 9]]
    points = [[0],
              [-1 / np.sqrt(3), 1 / np.sqrt(3)],
              [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]]

    gauss_wgh = weights[scheme - 1]
    gauss_loc = points[scheme - 1]

    result = 0
    for xi, wx in zip(gauss_loc, gauss_wgh):
        for eta, wy in zip(gauss_loc, gauss_wgh):
            result += function(xi, eta) * wx * wy
    return result


def gauss_1D(function, scheme=3):
    weights = [[2],
               [1, 1],
               [5 / 9, 8 / 9, 5 / 9]]
    points = [[0],
              [-1 / np.sqrt(3), 1 / np.sqrt(3)],
              [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]]

    gauss_wgh = weights[scheme - 1]
    gauss_loc = points[scheme - 1]

    result = 0
    for xi, wx in zip(gauss_loc, gauss_wgh):
        result += function(xi) * wx

    return result
