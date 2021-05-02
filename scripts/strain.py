import numpy as np


def engineering_strain(gradient):
    return 0.5 * (gradient + gradient.T) - np.identity(3)


def green_lagrange(gradient):
    return 0.5 * (np.dot(gradient.T, gradient) - np.identity(3))


def green_lagrange_rate(gradient, gradient_rate):
    mult = np.dot(gradient_rate.T, gradient)
    return 0.5 * (mult + mult.T)
