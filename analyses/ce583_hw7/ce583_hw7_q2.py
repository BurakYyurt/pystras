import numpy as np
import pickle
import preprocess.ce583_hw7_q2_8x1 as ce583_hw7
from scripts import model, strain, stress
from scripts import deformation_gradient as df
import matplotlib.pyplot as plt

name = "583_q3_8x2"
fea_model = model.Model(name, ce583_hw7)
fea_model.assemble()
fea_model.solve()

cnt = 0
mesh_cnt = 2
str_out = np.zeros((100 + mesh_cnt, 3))
coords = np.zeros((100 + mesh_cnt, 2))

print(fea_model.U, fea_model.R)

