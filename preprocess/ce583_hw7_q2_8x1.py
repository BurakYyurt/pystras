import numpy as np
from scripts.ElementLibraries import Mindlin as fel

# ---------------------------------------------- Model Creation ------------------------------------------------------ #
# This module creates a cantilever from bricks with 8 corner nodes.
# Boundary conditions, DOF matrix, Connectivity Matrix, Nodal Coordinate Matrix are obtained in this file.


# Inputs
# Geometry
dimension_x = 4000  # mm, length
dimension_y = 600  # mm, height
dimension_z = 400  # mm, thickness

division_x = 8  # division count in x direction
division_z = 1  # division count in z direction

# Materials
E = 25000  # Young's Modulus N/mm2
v = 0.0  # Poisson's Ratio

# Prescribed displacements at nodes
boundary = [{"node_id": 0, "y": 0, "theta_z": 0},
            {"node_id": 1, "y": 0, "theta_z": 0}]

# Prescribed forces at nodes
nodal_forces = [{"node_id": 16, "y": -10000},
                {"node_id": 17, "y": -10000}]

# Prescribed forces at edges
# TODO
# Prescribed forces at areas
# TODO
# Prescribed body forces
# TO BE IMPROVED
body_forces = -10

grid_x = division_x + 1  # grid count in x direction
grid_z = division_z + 1  # grid count in z direction

increment_x = dimension_x / division_x
increment_z = dimension_z / division_z

n_quad = division_x * division_z  # number of quads
n_node = (division_x + 1) * (division_z + 1)  # number of nodes
n_dof = n_node * 3  # number of DOF's

coordinates = np.zeros((n_node, 2))

x = 0
z = 0
node = 0
grids = np.zeros((grid_x, grid_z))

for i in range(grid_x):
    for j in range(grid_z):
        coordinates[node] = (x, z)
        grids[i, j] = node
        node += 1
        z += increment_z
    x += increment_x
    z = 0

connectivity = np.zeros((n_quad, 4), dtype=int)
member = 0

for i in range(division_x):
    for j in range(division_z):
        node = grids[i, j]
        connectivity[member] = (node, grid_z + node, grid_z + node + 1, node + 1)
        member += 1

dof_matrix = np.ones((n_node, 3)) * -1

n = n_dof - 1
for i in boundary:
    node_id = i["node_id"]
    if "theta_z" in i:
        dof_matrix[node_id, 2] = n
        n -= 1
    if "y" in i:
        dof_matrix[node_id, 0] = n
        n -= 1
n_boundary = n_dof - n
n = 0

for k, i in enumerate(dof_matrix):
    for t, j in enumerate(i):
        if j < 0:
            dof_matrix[k, t] = n
            n += 1
dof_matrix = dof_matrix.astype('int')

f_node = np.zeros(n_dof)
for i in nodal_forces:
    node_id = i["node_id"]
    if "theta_z" in i:
        dof_id = dof_matrix[node_id, 2]
        f_node[dof_id] = i["theta_z"]
    if "y" in i:
        dof_id = dof_matrix[node_id, 0]
        f_node[dof_id] = i["y"]

u_node = np.zeros(n_dof)
for i in boundary:
    node_id = i["node_id"]
    if "y" in i:
        dof_id = dof_matrix[node_id, 0]
        u_node[dof_id] = i["y"]
    if "theta_z" in i:
        dof_id = dof_matrix[node_id, 2]
        u_node[dof_id] = i["theta_z"]

members = [fel.element(connectivity[i], coordinates[connectivity[i]],
                       dof_matrix[connectivity[i]], 3) for i in range(n_quad)]
for i in members:
    i.element_properties(dimension_y)
    i.material_properties(E, v, 0)
    i.forces(np.zeros(3), np.zeros(2), 0)
