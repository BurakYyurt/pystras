import numpy as np


def deformation_gradient(element, x0, U, location):
    xi, eta, zeta = location
    rho0 = element.rho0
    xt = x0 + U
    _, _, _, _, J0 = element.shape_functions(xi, eta, zeta, x0)
    _, _, _, _, Jt = element.shape_functions(xi, eta, zeta, xt)
    X_0_t = (np.dot(np.linalg.inv(J0), Jt)).T
    rhot = rho0 / np.linalg.det(X_0_t)

    return X_0_t, rhot


def deformation_gradient_rate(gradient_tdt, gradient_t, dt):
    return (gradient_tdt - gradient_t) / dt
