import numpy as np
import pickle
from scripts import assemble, solve


class Model:
    def __init__(self, name, preprocess):
        self.name = name
        self.members = preprocess.members  # List containing all FEM member information
        self.connectivity = preprocess.connectivity  # Connectivity matrix of the model
        self.coordinates = preprocess.coordinates  # Nodal coordinates of the model
        self.dof_matrix = preprocess.dof_matrix  # Matrix relating DOF ID's to Joint ID's.
        self.n_boundary = preprocess.n_boundary  # Number of restrained boundaries
        self.f_node = preprocess.f_node  # Prescribed Nodal Forces
        self.u_node = preprocess.u_node  # Prescribed Nodal Displacements
        self.K = []  # Global Stiffness Matrix
        self.F = []  # Global Force Vector
        self.U = []  # Nodal Displacements
        self.R = []  # Nodal Reactions

    def assemble(self):
        self.K, self.F = assemble.assemble(self.members, self.dof_matrix)

    def solve(self):
        self.U, self.R = solve.linear_solver(self.K, self.F, self.f_node, self.u_node, self.n_boundary)
        return self.U

    def dump(self):
        file = open(self.name + "members", "wb")
        pickle.dump(self.members, file)
        shape = self.dof_matrix.shape[1]
        np.savetxt("Disp" + self.name + ".csv",
                   self.U.reshape(int(len(self.U) / shape), shape), delimiter=";")
        np.savetxt("React" + self.name + ".csv",
                   self.R.reshape(int(len(self.R) / shape), shape), delimiter=";")
