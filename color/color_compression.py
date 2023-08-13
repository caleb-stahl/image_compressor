import numpy as np
import numpy.linalg as npla

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

image = plt.imread('HG.jpg') # Make sure the image you want to compress is in the same directory
mat = image.copy()
plt.figure(figsize=(10,10))
plt.imshow(mat)
mat_red = mat[:,:,0]
mat_green = mat[:,:,1]
mat_blue = mat[:,:,2]
num_row, num_col = mat_red.shape
compression = abs(float(input("Enter Compression Factor: ")))
assert(compression != 0)
org_stor = num_row * num_col
com_stor = org_stor / compression
k = com_stor / ((num_col + num_row))
u_r, s_r, v_r = np.linalg.svd(mat_red)
u_g, s_g, v_g = np.linalg.svd(mat_green)
u_b, s_b, v_b = np.linalg.svd(mat_blue)
mr1 = np.zeros(mat_red.shape)
mg1 = np.zeros(mat_red.shape)
mb1 = np.zeros(mat_red.shape)

for i in range(int(k)):
    mr1 += s_r[i] * np.outer(u_r[:, i], v_r[i, :])
    mg1 += s_g[i] * np.outer(u_g[:, i], v_g[i, :])
    mb1 += s_b[i] * np.outer(u_b[:, i], v_b[i, :])

final = np.stack((mr1, mg1, mb1), axis=2)
final = final/255.0
plt.figure(figsize=(10,10))
plt.imshow(final)
plt.show()
print()