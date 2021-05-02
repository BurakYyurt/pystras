import numpy as np
import pickle
import preprocess.ce583_hw6_q2_20x5 as ce583_hw6
from scripts import model, strain, stress
from scripts import deformation_gradient as df
import matplotlib.pyplot as plt

name = "583_q2_20x5"
fea_model = model.Model(name, ce583_hw6)
fea_model.assemble()
fea_model.solve()

cnt = 0
mesh_cnt = 5
str_out = np.zeros((100 + mesh_cnt, 3))
coords = np.zeros((100 + mesh_cnt, 2))

start = int(-100 / mesh_cnt / 2)
end = int(100 / mesh_cnt / 2) + 1

for i in range(mesh_cnt):
    for j in range(start, end):
        member = fea_model.members[i]
        dofs = member.dofs
        U = fea_model.U[dofs]
        eps = member.get_strain(U, [-1, j / start])
        stress_val = stress.engineering_stress(eps, member.C)
        N, _, _, _ = member.shape_functions(-1, j / start)
        x, y = np.dot(N, member.coordinates)
        str_out[cnt] = stress_val[:]
        coords[cnt] = [x, y]
        cnt += 1

np.savetxt("stress_20x5.csv", str_out, delimiter=";")
np.savetxt("coordinates.csv", coords, delimiter=";")

# plt.scatter(str_out[:, 2], coords[:, 1], s=0.1)
# plt.show()

selected_members = [0, 1, 2, 3, 4, 45, 46, 47, 48, 49, 95, 96, 97, 98, 99]
locations = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cnt = 0
str_out = np.zeros((15, 3))
coords = np.zeros((15, 2))

for n, i in enumerate(selected_members):
    member = fea_model.members[i]
    dofs = member.dofs
    U = fea_model.U[dofs]
    eps = member.get_strain(U, [locations[n], 0])
    stress_val = stress.engineering_stress(eps, member.C)
    N, _, _, _ = member.shape_functions(locations[n], 0)
    x, y = np.dot(N, member.coordinates)
    str_out[cnt] = stress_val[:]
    coords[cnt] = [x, y]
    cnt += 1

np.savetxt("stress_20x5_2b.csv", str_out, delimiter=";")
np.savetxt("coordinates_2b.csv", coords, delimiter=";")

fea_model.dump()
