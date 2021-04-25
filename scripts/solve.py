import numpy as np


def linear_solver(K, F, F_node, U_node, n_boundary):
    n_dof = K.shape[0]
    n_free_dof = n_dof - n_boundary + 1
    Kff = K[:n_free_dof, :n_free_dof]
    Kfr = K[:n_free_dof, n_free_dof:]
    Krf = K[n_free_dof:, :n_free_dof]
    Krr = K[n_free_dof:, n_free_dof:]
    Ff = F[:n_free_dof] + F_node[:n_free_dof]
    Fr = F[n_free_dof:] + F_node[n_free_dof:]
    Ur = U_node[n_free_dof:]
    Uf = np.dot(np.linalg.inv(Kff), (Ff - np.dot(Kfr, Ur)))
    R = np.dot(Krf, Uf) + np.dot(Krr, Ur) - Fr
    U = np.zeros(n_dof)
    U[:n_free_dof] = Uf
    U[n_free_dof:] = Ur

    return U, R
