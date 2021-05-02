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
dU = ce522_hw4.deltaU.flatten()
X_t_tdt, rho_t_tdt = df.deformation_gradient(member, xt, dU, location)
X_0_tdt = np.dot(X_0_t, X_t_tdt)

Eps_0_tdt = strain.green_lagrange(X_0_tdt)

e_t = strain.engineering_strain(X_t_tdt)
e_0 = np.dot(np.dot(X_0_t.T, e_t), X_0_t)

dt = 1

Xdot_0_t = df.deformation_gradient_rate(member, x0, Ut, 10, location)
Epsdot_0_t = strain.green_lagrange_rate(X_0_t, Xdot_0_t)

X_t_0 = np.linalg.inv(X_0_t)
v_t = np.dot(np.dot(X_t_0.T, Epsdot_0_t), X_t_0)

print("X_0_t:")
print(X_0_t)
print("X_t_tdt:")
print(X_t_tdt)
print("X_0_tdt:")
print(X_0_tdt)
print("rho_t:")
print(rho_t)
print("Eps_0_t")
print(Eps_0_t)
print("Eps_0_tdt")
print(Eps_0_tdt)
print("e_t")
print(e_t)
print("e_0")
print(e_0)
print("v_t")
print(v_t)
