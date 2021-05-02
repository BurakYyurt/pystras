import numpy as np
import pickle
import preprocess.ce583_hw6_q3 as ce583_hw6
from scripts import model, strain, stress
from scripts import deformation_gradient as df
import matplotlib.pyplot as plt

name = "583_q3_8x2"
fea_model = model.Model(name, ce583_hw6)
fea_model.assemble()
fea_model.solve()

cnt = 0
mesh_cnt = 2
str_out = np.zeros((100 + mesh_cnt, 3))
coords = np.zeros((100 + mesh_cnt, 2))

start = int(-100 / mesh_cnt / 2)
end = int(100 / mesh_cnt / 2) + 1

for i in range(mesh_cnt):
    for j in range(start, end):
        member = fea_model.members[i]
        dofs = member.dofs
        U = fea_model.U[dofs]
        eps = member.get_strain(U, [0, j / start])
        stress_val = stress.engineering_stress(eps, member.C)
        N, _, _, _ = member.shape_functions(0, j / start)
        x, y = np.dot(N, member.coordinates)
        str_out[cnt] = stress_val[:]
        coords[cnt] = [x, y]
        cnt += 1

# plt.plot(str_out[:, 2], coords[:, 1])
# plt.show()

np.savetxt("stress_8x2.csv", str_out, delimiter=";")
np.savetxt("coordinates.csv", coords, delimiter=";")

selected_members = [0, 0, 2, 1]

nodes = np.array([[[1, -1], [1, 1]],
                  [[1, 1], [-1, 1]],
                  [[-1, -1], [-1, 1]],
                  [[1, -1], [-1, -1]]])

direction = [0, 1, 0, 1]
steps = 10
Na = np.zeros((2, 4))
disps = np.zeros((4, steps+1))

for k, i in enumerate(selected_members):
    xi_start = nodes[k, 0, 0]
    eta_start = nodes[k, 0, 1]
    xi_end = nodes[k, 1, 0]
    eta_end = nodes[k, 1, 1]
    xi_increment = (xi_end - xi_start) / steps
    eta_increment = (eta_end - eta_start) / steps

    member = fea_model.members[i]
    dofs = member.dofs
    U = fea_model.U[dofs]
    _ = member.get_strain(U, [0, 0])
    u_node = member.u_member[:8]
    u_gen = member.u_member[8:]

    xi = xi_start
    eta = eta_start
    print("next")
    for j in range(steps+1):
        print(xi,eta)
        xi2 = 1 - xi ** 2
        eta2 = 1 - eta ** 2
        Na[0, 0] = xi2
        Na[0, 1] = eta2
        Na[1, 2] = xi2
        Na[1, 3] = eta2
        N, _, _, _ = member.shape_functions(xi, eta)
        u = np.dot(N, u_node) + np.dot(Na, u_gen)
        disps[k, j] = u[direction[k]]
        xi += xi_increment
        eta += eta_increment

np.savetxt("edges.csv", disps, delimiter=";")

fea_model.dump()
