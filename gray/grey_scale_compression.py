import numpy as np
import numpy.linalg as npla

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d



np.set_printoptions(precision = 4)


image = plt.imread("hg.jpg") #Make sure the image you want to compress is in the same directory
mat = np.float64(image[:,:,0])
num_row, num_col = mat.shape
U, sigma, Vt = npla.svd(mat, full_matrices=False)
plt.figure(figsize=(10,10))
plt.gray()
plt.imshow(mat)
plt.title('original image, %d by %d pixels' % (num_row, num_col))
print()

#Compressing the Image
compression = abs(float(input("Enter Compression Factor: ")))
assert(compression != 0)
org_stor = num_row * num_col
com_stor = org_stor / compression
k = com_stor / (num_col + num_row)
mat2 = np.zeros(mat.shape)

#Reconstructing the Image
for i in range(int(k)):
    mat2 += sigma[i] * np.outer(U[:,i], Vt[i,:])

#Plotting the Image
plt.figure(figsize=(10,10))
plt.gray()
plt.imshow(mat2)
plt.title('compressed image, compression factor %d' % compression)
plt.show()
print()
