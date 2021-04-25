import numpy as np


# def engineering_strain(member, U, location):
#     u_elem = U[member.dofs]
#     xi, eta = location
#     B, detJ = member.strain_displacement(xi, eta)
#     return np.dot(B, u_elem)


def engineering_strain(gradient):
    return 0.5 * (gradient + gradient.T)


def green_lagrange(gradient):
    return 0.5 * (np.dot(gradient.T, gradient) - np.identity(3))


