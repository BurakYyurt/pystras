import numpy as np
import pickle
import preprocess.ce583_hw6_1_8x1 as ce583_hw6
from scripts import model, strain, stress
from scripts import deformation_gradient as df

name = "583_q1_8x1"
fea_model = model.Model(name, ce583_hw6)
fea_model.assemble()
fea_model.solve()

str_out = np.zeros((1001, 2))
cnt = 0
for i in range(1):
    for j in range(-500, 501):
        member = fea_model.members[i]
        eps = strain.engineering_strain(member, fea_model.U, [-1, j / 250])
        stress_val = stress.engineering_stress(eps, member.C)
        N, _, _, _ = member.shape_functions(-1, j / 250)
        x, y = np.dot(N, member.coordinates)
        str_out[cnt] = [stress_val[0], y]
        cnt += 1

    np.savetxt("stress_8x1.csv", str_out, delimiter=",")
fea_model.dump()
