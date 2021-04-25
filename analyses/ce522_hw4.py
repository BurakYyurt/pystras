import numpy as np
import pickle
import preprocess.ce522_hw4 as ce522_hw4
from scripts import model, strain, stress
from scripts import deformation_gradient as df

name = "522"
fea_model = model.Model(name, ce522_hw4)

member = fea_model.members[0]

Ut = ce522_hw4.Ut.flatten()
x0 = member.coordinates

location = [0, 0, 0]
X_0_t, rho_t = df.deformation_gradient(member, x0, Ut, location)
Eps_0_t = strain.green_lagrange(X_0_t)

xt = x0 + Ut
dt = ce522_hw4.deltaU.flatten()
X_t_tdt, rho_t_tdt = df.deformation_gradient(member, xt, dt, location)
Eps_0_tdt = strain.green_lagrange(X_0_t * X_t_tdt)

Xdot_t_tdt = df.deformation_gradient_rate(X_t_tdt, X_0_t, dt=1)

e_t = strain.engineering_strain(X_0_t)
e_0 = np.dot(np.dot(X_0_t.T, e_t), X_0_t)

print(X_0_t)

# str_out = np.zeros((1002, 2))
# cnt = 0
# for i in range(2):
#     for j in range(-250, 251):
#         member = fea_model.members[i]
#         eps = strain.engineering_strain(member, fea_model.U, [-1, j / 250])
#         stress_val = stress.engineering_stress(eps, member.C)
#         N, _, _, _ = member.shape_functions(-1, j/250)
#         x, y = np.dot(N, member.coordinates)
#         str_out[cnt] = [stress_val[0], y]
#         cnt += 1
#
#     np.savetxt("stress_8x2.csv", str_out, delimiter=",")
# fea_model.dump()


