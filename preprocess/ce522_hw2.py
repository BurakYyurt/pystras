import numpy as np
from scripts import element_library as fel

# ---------------------------------------------- Model Creation ------------------------------------------------------ #
# This module creates a cantilever from bricks with 8 corner nodes.
# Boundary conditions, DOF matrix, Connectivity Matrix, Nodal Coordinate Matrix are obtained in this file.


# Inputs
# Geometry
dimension_x = 300  # mm, length
dimension_y = 20  # mm, width
dimension_z = 20  # mm, height

division_x = 15  # division count in x direction
division_y = 1  # division count in y direction
division_z = 1  # division count in z direction

# Materials
E = 200000  # Young's Modulus N/mm2
v = 0.3  # Poisson's Ratio

# Prescribed displacements at nodes
boundary = [{"node_id": 0, "x": 0, "y": 0, "z": 0},
            {"node_id": 1, "x": 0, "y": 0, "z": 0},
            {"node_id": 2, "x": 0, "y": 0, "z": 0},
            {"node_id": 3, "x": 0, "y": 0, "z": 0}]  # ,
# {"node_id": 4, "x": 0, "y": 0, "z": 0},
# {"node_id": 5, "x": 0, "y": 0, "z": 0},
# {"node_id": 6, "x": 0, "y": 0, "z": 0},
# {"node_id": 7, "x": 0, "y": 0, "z": 0},
# {"node_id": 8, "x": 0, "y": 0, "z": 0}]

# Prescribed forces at nodes
nodal_forces = [{"node_id": 60, "z": -25000},
                {"node_id": 61, "z": -25000},
                {"node_id": 62, "z": -25000},
                {"node_id": 63, "z": -25000}]  # ,
# {"node_id": 141, "z": -25000}]

# Prescribed forces at edges
# TODO
# Prescribed forces at areas
# TODO
# Prescribed body forces
# TO BE IMPROVED
body_forces = np.array([0,0,-10])

grid_x = division_x + 1  # grid count in x direction
grid_y = division_y + 1  # grid count in y direction
grid_z = division_z + 1  # grid count in z direction

increment_x = dimension_x / division_x
increment_y = dimension_y / division_y
increment_z = dimension_z / division_z

n_brick = division_x * division_y * division_z  # number of bricks
n_node = (division_x + 1) * (division_y + 1) * (division_z + 1)  # number of bricks
n_dof = n_node * 3  # number of DOF's

coordinates = np.zeros((n_node, 3))

x = 0
y = 0
z = 0
node = 0
grids = np.zeros((grid_x, grid_y, grid_z))

for i in range(grid_x):
    for j in range(grid_y):
        for k in range(grid_z):
            coordinates[node] = (x, y, z)
            grids[i, j, k] = node
            node += 1
            z += increment_z
        y += increment_y
        z = 0
    x += increment_x
    y = 0

connectivity = np.zeros((n_brick, 8), dtype=int)
member = 0

for i in range(division_x):
    for j in range(division_y):
        for k in range(division_z):
            node = grids[i, j, k]
            connectivity[member] = (node, grid_z * grid_y + node,
                                    grid_z * (grid_y + 1) + node, node + grid_z,
                                    node + 1, grid_z * grid_y + node + 1,
                                    grid_z * (grid_y + 1) + node + 1, node + grid_z + 1)
            member += 1

dof_matrix = np.ones((n_node, 3)) * -1

n = n_dof - 1
for i in boundary:
    node_id = i["node_id"]
    if "z" in i:
        dof_matrix[node_id, 2] = n
        n -= 1
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
    if "z" in i:
        dof_id = dof_matrix[node_id, 2]
        f_node[dof_id] = i["z"]

u_node = np.zeros(n_dof)
for i in boundary:
    node_id = i["node_id"]
    if "x" in i:
        dof_id = dof_matrix[node_id, 0]
        u_node[dof_id] = i["x"]
    if "y" in i:
        dof_id = dof_matrix[node_id, 1]
        u_node[dof_id] = i["y"]
    if "z" in i:
        dof_id = dof_matrix[node_id, 2]
        u_node[dof_id] = i["z"]

members = [fel.BrickBilinear(connectivity[i], coordinates[connectivity[i]],
                             dof_matrix[connectivity[i]], 3) for i in range(n_brick)]
for i in members:
    i.material_properties(E, v, 0)
    i.forces(body_forces, np.zeros(3), 0)
