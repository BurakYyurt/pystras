import numpy as np
from scripts import integral


class BrickBilinear:
    def __init__(self, nodes, coordinates, dofs, integration):
        self.nodes = nodes
        self.coordinates = coordinates.flatten()
        self.dofs = dofs.flatten()
        self.integral = integral.gauss_3D
        self.integration_points = integration

    def material_properties(self, E, v, rho):
        self.E = E
        self.v = v
        self.rho0 = rho
        self.C = (E / (1 + v) / (1 - 2 * v)) * np.array([[1 - v, v, v, 0, 0, 0],
                                                         [v, 1 - v, v, 0, 0, 0],
                                                         [v, v, 1 - v, 0, 0, 0],
                                                         [0, 0, 0, (1 - 2 * v) / 2, 0, 0],
                                                         [0, 0, 0, 0, (1 - 2 * v) / 2, 0],
                                                         [0, 0, 0, 0, 0, (1 - 2 * v) / 2]])

    def forces(self, f_body, f_surface, f_edge):
        self.f_body = f_body
        self.f_surface = f_surface
        self.f_edge = f_edge

    def shape_functions(self, xi, eta, zeta, coordinates=False):
        if not coordinates.any():
            coordinates = self.coordinates
        xi_ = [-1, -1, -1, -1, 1, 1, 1, 1]
        eta_ = [-1, -1, 1, 1, -1, -1, 1, 1]
        zeta_ = [1, -1, -1, 1, 1, -1, -1, 1]

        N = np.zeros((3, 24))
        dNdxi = np.zeros((3, 24))
        dNdeta = np.zeros((3, 24))
        dNdzeta = np.zeros((3, 24))
        J = np.zeros((3, 3))

        for i in range(8):
            for j in range(3):
                N[j, i * 3 + j] = (1 / 8) * (1 + xi_[i] * xi) * (1 + eta_[i] * eta) * (1 + zeta_[i] * zeta)
                dNdxi[j, i * 3 + j] = (1 / 8) * xi_[i] * (1 + eta_[i] * eta) * (1 + zeta_[i] * zeta)
                dNdeta[j, i * 3 + j] = (1 / 8) * (1 + xi_[i] * xi) * eta_[i] * (1 + zeta_[i] * zeta)
                dNdzeta[j, i * 3 + j] = (1 / 8) * (1 + xi_[i] * xi) * (1 + eta_[i] * eta) * zeta_[i]

        dXdxi = np.dot(dNdxi, coordinates)
        dXdeta = np.dot(dNdeta, coordinates)
        dXdzeta = np.dot(dNdzeta, coordinates)
        J[[0], :] = dXdxi.T
        J[[1], :] = dXdeta.T
        J[[2], :] = dXdzeta.T

        return N, dNdxi, dNdeta, dNdzeta, J

    def strain_displacement(self, xi, eta, zeta):
        N, dNdxi, dNdeta, dNdzeta, J = self.shape_functions(xi, eta, zeta)

        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        S = np.zeros((6, 9))
        G = np.zeros((9, 24))
        A = np.zeros((9, 9))

        S[0, 0] = 1
        S[1, 4] = 1
        S[2, 8] = 1
        S[3, 1] = 1
        S[3, 3] = 1
        S[4, 2] = 1
        S[4, 6] = 1
        S[5, 5] = 1
        S[5, 7] = 1

        A[0:3, 0:3] = invJ
        A[3:6, 3:6] = invJ
        A[6:, 6:] = invJ

        G[[0], :] = dNdxi[0]
        G[[1], :] = dNdeta[0]
        G[[2], :] = dNdzeta[0]
        G[[3], :] = dNdxi[1]
        G[[4], :] = dNdeta[1]
        G[[5], :] = dNdzeta[1]
        G[[6], :] = dNdxi[2]
        G[[7], :] = dNdeta[2]
        G[[8], :] = dNdzeta[2]

        B = np.dot(np.dot(S, A), G)
        print(B)
        return B, detJ

    def stiffness(self, xi, eta, zeta):
        B, detJ = self.strain_displacement(xi, eta, zeta)
        return np.dot(np.dot(B.T, self.C), B) * detJ

    def f_element_body(self, xi, eta, zeta):
        N, _, _, _, J = self.shape_functions(xi, eta, zeta)
        detJ = np.linalg.det(J)
        return np.dot(N.T, self.f_body) * detJ


class QuadBilinear:
    def __init__(self, nodes, coordinates, dofs, integration):
        self.nodes = nodes
        self.coordinates = coordinates.flatten()
        self.dofs = dofs.flatten()
        self.integral = integral.gauss_2D
        self.integration_points = integration

    def element_properties(self, thickness, membrane_type="PlaneStress"):
        self.membrane_type = membrane_type
        self.t = thickness

    def material_properties(self, E, v, rho):
        self.E = E
        self.v = v
        self.rho = rho
        if self.membrane_type == "PlaneStrain":
            self.C = (E / (1 - 2 * v) / (1 + v)) * np.array([[1 - v, v, 0],
                                                             [v, 1 - v, 0],
                                                             [0, 0, (1 - 2 * v) / 2]])

        elif self.membrane_type == "PlaneStress":
            self.C = (E / (1 - v ** 2)) * np.array([[1, v, 0],
                                                    [v, 1, 0],
                                                    [0, 0, (1 - v) / 2]])

    def forces(self, f_body, f_surface, f_edge):
        self.f_body = f_body
        self.f_surface = f_surface
        self.f_edge = f_edge

    def shape_functions(self, xi, eta):
        xi_ = [-1, 1, 1, -1]
        eta_ = [-1, -1, 1, 1]

        N = np.zeros((2, 8))
        dNdxi = np.zeros((2, 8))
        dNdeta = np.zeros((2, 8))
        J = np.zeros((2, 2))

        for i in range(4):
            for j in range(2):
                N[j, i * 2 + j] = (1 / 4) * (1 + xi_[i] * xi) * (1 + eta_[i] * eta)
                dNdxi[j, i * 2 + j] = (1 / 4) * xi_[i] * (1 + eta_[i] * eta)
                dNdeta[j, i * 2 + j] = (1 / 4) * (1 + xi_[i] * xi) * eta_[i]

        dXdxi = np.dot(dNdxi, self.coordinates)
        dXdeta = np.dot(dNdeta, self.coordinates)
        J[[0], :] = dXdxi.T
        J[[1], :] = dXdeta.T

        return N, dNdxi, dNdeta, J

    def strain_displacement(self, xi, eta):
        N, dNdxi, dNdeta, J = self.shape_functions(xi, eta)

        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        S = np.zeros((3, 4))
        G = np.zeros((4, 8))
        A = np.zeros((4, 4))

        S[0, 0] = 1
        S[1, 3] = 1
        S[2, 1] = 1
        S[2, 2] = 1

        A[0:2, 0:2] = invJ
        A[2:4, 2:4] = invJ

        G[[0], :] = dNdxi[0]
        G[[1], :] = dNdeta[0]
        G[[2], :] = dNdxi[1]
        G[[3], :] = dNdeta[1]

        B = np.dot(np.dot(S, A), G)

        return B, detJ

    def stiffness(self, xi, eta):
        B, detJ = self.strain_displacement(xi, eta)
        return np.dot(np.dot(B.T, self.C), B) * detJ * self.t

    def f_element_body(self, xi, eta):
        N, _, _, J = self.shape_functions(xi, eta)
        detJ = np.linalg.det(J)
        return np.dot(N.T, self.f_body) * detJ


class QuadBilinear_Incompatible:
    pass