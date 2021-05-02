import numpy as np
import pickle
import preprocess.ce583_hw6_q2_8x1 as ce583_hw6
from scripts import model, strain, stress
from scripts import deformation_gradient as df

name = "583_q2_8x1"
fea_model = model.Model(name, ce583_hw6)
fea_model.assemble()
fea_model.solve()

cnt = 0
mesh_cnt = 1
str_out = np.zeros((100+mesh_cnt, 3))
coords = np.zeros((100+mesh_cnt, 2))

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

np.savetxt("stress_8x1.csv", str_out, delimiter=";")
np.savetxt("coordinates.csv", coords, delimiter=";")

fea_model.dump()
