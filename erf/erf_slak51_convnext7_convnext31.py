# A script to visualize the ERF.
# More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity (https://arxiv.org/pdf/2207.03620.pdf)
# Github source: https://github.com/VITA-Group/SLaK
# Licensed under The MIT License [see LICENSE for details]
# Modified from https://github.com/DingXiaoH/RepLKNet-pytorch.
# --------------------------------------------------------'
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import pyplot as plt

fig = figure(num=None, figsize=(12, 4), dpi=150, facecolor='w', edgecolor='k')


matrix1 = np.load('erf/Saved_matrix/ConvNext-T_7.npy')
matrix2 = np.load('erf/Saved_matrix/ConvNext-T_31.npy')
matrix3 = np.load('erf/Saved_matrix/SLaK-T_51.npy')

matrice = [matrix1, matrix2, matrix3]

for i in range(len(matrice)):
    # matrice[i] = np.log10(matrice[i] + 1)       #  do not need log, which will decrease the rea
    matrice[i] = matrice[i] / np.max(matrice[i])  # scale


erf1 = fig.add_subplot(1,3,1)
im1 = erf1.matshow(matrice[0],  cmap='pink', interpolation='nearest')
erf1.set_title('ConvNeXt [7, 7, 7, 7]', fontsize=10, y=1.12)
divider = make_axes_locatable(erf1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)


erf2 = fig.add_subplot(1,3,2)
im2 = erf2.matshow(matrice[1], cmap='pink', interpolation='nearest' )
erf2.set_title('ConvNeXt (RepLKNet) [31, 29, 27, 13]', fontsize=10, y=1.12)
divider = make_axes_locatable(erf2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)


erf3 = fig.add_subplot(1,3,3)
im3 = erf3.matshow(matrice[2], cmap='pink', interpolation='nearest' )
erf3.set_title('SLaK [51, 49, 47, 13]', fontsize=10, y=1.12)
divider = make_axes_locatable(erf3)
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)
plt.tight_layout()

plt.show()