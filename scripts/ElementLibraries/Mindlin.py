import numpy as np
from scripts import integral


class element:
    def __init__(self, nodes, coordinates, dofs, integration):
        self.nodes = nodes
        self.coordinates = coordinates.flatten()
        self.dofs = dofs.flatten()
        self.integral = integral.gauss_2D
        self.integration_points = integration

    def element_properties(self, thickness, membrane_type="ThickPlate"):
        self.membrane_type = membrane_type
        self.t = thickness

    def material_properties(self, E, v, rho):
        self.E = E
        self.v = v
        self.rho = rho

        self.Cb = (E / (1 - v ** 2)) * np.array([[1, v, 0],
                                                 [v, 1, 0],
                                                 [0, 0, (1 - v) / 2]])

        self.Cs = E / 2 / (1 + v) * np.identity(2)

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
        invJ = np.linalg.inv(J)

        dNdX = np.dot(invJ, np.array([dNdxi[0], dNdeta[0]]))

        return N, dNdX, J

    def strain_displacement(self, xi, eta):
        N, dNdX, J = self.shape_functions(xi, eta)

        detJ = np.linalg.det(J)
        Bb = np.zeros((3, 12))
        Bs = np.zeros((2, 12))

        for i in range(4):
            Bb[0, i * 3 + 2] = dNdX[0, 2 * i]
            Bb[1, i * 3 + 1] = -dNdX[1, 2 * i]
            Bb[2, i * 3 + 1] = -dNdX[0, 2 * i]
            Bb[2, i * 3 + 2] = dNdX[1, 2 * i]
            Bs[0, i * 3] = dNdX[0, 2 * i]
            Bs[0, i * 3 + 2] = N[0, 2 * i]
            Bs[1, i * 3 + 1] = N[0, 2 * i]
            Bs[1, i * 3] = dNdX[1, 2 * i]

        return Bb, Bs, detJ

    def stiffness_bending(self, xi, eta):
        Bb, _, detJ = self.strain_displacement(xi, eta)
        return np.dot(np.dot(Bb.T, self.Cb), Bb) * detJ * (self.t ** 3) / 12

    def stiffness_shear(self, xi, eta):
        _, Bs, detJ = self.strain_displacement(xi, eta)
        return np.dot(np.dot(Bs.T, self.Cs), Bs) * detJ * self.t * 5 / 6

    def force(self, xi, eta):
        N, _, J = self.shape_functions(xi, eta)
        detJ = np.linalg.det(J)
        force = np.zeros(12)
        cnt = 0
        for i in N.T[::2]:
            for j in self.f_body:
                force[cnt] = i[0] * j
                cnt+=1

        return force

    def integrate(self):
        k_bending = self.integral(self.stiffness_bending)
        k_shear = self.integral(self.stiffness_shear, scheme=1)
        f_elem = self.integral(self.force)
        return k_shear + k_bending, f_elem
