from scipy import ndimage, spatial
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from ddw.utils.mrctools2 import load_data, save_mrc_data

a = load_data("testing/tomos/tomo0/smiley/0.pt").float()
b = np.zeros((1280,1280))
c = ndimage.rotate(a, angle=60, reshape= False, mode= 'reflect')

# save_mrc_data(a, "testrotate_A.mrc")
# # save_mrc_data(b, "testrotate_B.mrc")
# save_mrc_data(c, "testrotate_C.mrc")


plt.imshow(a)
plt.imshow(c)

plt.show()