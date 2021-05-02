import numpy as np
from scripts.ElementLibraries import brickBilinear as fel

# ---------------------------------------------- Model Creation ------------------------------------------------------ #
# This module creates a cantilever from bricks with 8 corner nodes.
# Boundary conditions, DOF matrix, Connectivity Matrix, Nodal Coordinate Matrix are obtained in this file.


# Inputs
# Geometry
dimension_x = 4000  # mm, length
dimension_y = 600  # mm, height
dimension_z = 400  # mm, thickness

division_x = 8  # division count in x direction
division_y = 2  # division count in y direction

# Materials
E = 25000  # Young's Modulus N/mm2
v = 0.0  # Poisson's Ratio

# Prescribed displacements at nodes
boundary = [{"node_id": 0, "x": 0, "y": 0},
            {"node_id": 1, "x": 0, "y": 0},
            {"node_id": 2, "x": 0, "y": 0}]

# Prescribed forces at nodes
nodal_forces = [{"node_id": 26, "y": -20000}]

# Prescribed forces at edges
# TODO
# Prescribed forces at areas
# TODO
# Prescribed body forces
# TO BE IMPROVED
body_forces = 0

grid_x = division_x + 1  # grid count in x direction
grid_y = division_y + 1  # grid count in y direction

increment_x = dimension_x / division_x
increment_y = dimension_y / division_y

n_quad = division_x * division_y  # number of quads
n_node = (division_x + 1) * (division_y + 1)  # number of nodes
n_dof = n_node * 2  # number of DOF's

coordinates = np.zeros((n_node, 2))

x = 0
y = 0
node = 0
grids = np.zeros((grid_x, grid_y))

for i in range(grid_x):
    for j in range(grid_y):
        coordinates[node] = (x, y)
        grids[i, j] = node
        node += 1
        y += increment_y
    x += increment_x
    y = 0

connectivity = np.zeros((n_quad, 4), dtype=int)
member = 0

for i in range(division_x):
    for j in range(division_y):
        node = grids[i, j]
        connectivity[member] = (node, grid_y + node, grid_y + node + 1, node + 1)
        member += 1

dof_matrix = np.ones((n_node, 2)) * -1

n = n_dof - 1
for i in boundary:
    node_id = i["node_id"]
    if "y" in i:
        dof_matrix[node_id, 1] = n
        n -= 1
    if "x" in i:
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
    if "x" in i:
        dof_id = dof_matrix[node_id, 0]
        f_node[dof_id] = i["x"]
    if "y" in i:
        dof_id = dof_matrix[node_id, 1]
        f_node[dof_id] = i["y"]

u_node = np.zeros(n_dof)
for i in boundary:
    node_id = i["node_id"]
    if "x" in i:
        dof_id = dof_matrix[node_id, 0]
        u_node[dof_id] = i["x"]
    if "y" in i:
        dof_id = dof_matrix[node_id, 1]
        u_node[dof_id] = i["y"]

for j in range(3, 24):
    if j % 6 == 3 or j % 6 == 2:
        coordinates[j, 0] -= 100
    elif j % 6 == 5 or j % 6 == 0:
        coordinates[j, 0] += 100

members = [fel.QuadBilinearIncompatible(connectivity[i], coordinates[connectivity[i]],
                                        dof_matrix[connectivity[i]], 3) for i in range(n_quad)]
for i in members:
    i.element_properties(dimension_z)
    i.material_properties(E, v, 0)
    i.forces(np.zeros(2), np.zeros(2), 0)
