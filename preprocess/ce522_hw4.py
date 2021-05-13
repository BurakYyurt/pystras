import numpy as np
from scripts.ElementLibraries import brickBilinear as fel

E = 200000
v = 0.3
rho = 1
body_forces = [0, 0, 0]

coordinates = np.array([[200, 200, 400],
                        [200, 200, 200],
                        [200, 400, 200],
                        [200, 400, 400],
                        [400, 200, 400],
                        [400, 200, 200],
                        [400, 400, 200],
                        [400, 400, 400]])

Ut = np.array([[10, 10, 20],
               [10, 10, 10],
               [20, 20, 10],
               [20, 20, 20],
               [30, 40, 40],
               [30, 40, 20],
               [40, 80, 20],
               [40, 80, 40]])

deltaU = np.array([[1, 1, 2],
                   [1, 1, 1],
                   [2, 2, 1],
                   [2, 2, 2],
                   [3, 4, 4],
                   [3, 4, 2],
                   [4, 8, 2],
                   [4, 8, 4]])

connectivity = np.array([0, 1, 2, 3, 4, 5, 6, 7])
dof_matrix = np.array([[0, 1],
                       [2, 3],
                       [4, 5],
                       [6, 7],
                       [8, 9],
                       [10, 11],
                       [12, 13],
                       [14, 15]])
n_boundary = 0
f_node = 0
u_node = 0

members = [fel.element(connectivity, coordinates,
                             dof_matrix, 3)]
for i in members:
    i.material_properties(E, v, rho)
    i.forces(body_forces, np.zeros(3), 0)
