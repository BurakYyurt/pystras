import numpy as np
import pickle
import preprocess.ce583_hw5_8x1 as ce583_hw5
from scripts import model, strain, stress
from scripts import deformation_gradient as df
import matplotlib.pyplot as plt

name = "583_q2_8x1"
fea_model = model.Model(name, ce583_hw5)
fea_model.assemble()
fea_model.solve()

cnt = 0
mesh_cnt = 1
str_out = np.zeros((1001, 3))
coords = np.zeros((1001, 2))

start = int(-1000 / mesh_cnt / 2)
end = int(1000 / mesh_cnt / 2) + 1
#
# for i in range(mesh_cnt):
#     for j in range(start, end):
#         member = fea_model.members[i]
#         eps = strain.engineering_strain(member, fea_model.U, [-1, j / start])
#         stress_val = stress.engineering_stress(eps, member.C)
#         N, _, _, _ = member.shape_functions(-1, j / start)
#         x, y = np.dot(N, member.coordinates)
#         str_out[cnt] = stress_val[:]
#         coords[cnt] = [x, y]
#         cnt += 1
#
# plt.plot(str_out[:, 2], coords[:, 1])
# plt.show()
#
# print(stress_val)
# np.savetxt("stress_8x1.csv", str_out, delimiter=",")
# np.savetxt("coordinates.csv", coords, delimiter=",")
#
# fea_model.dump()
