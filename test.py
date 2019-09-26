from randomizedHoughEllipseDetection import FindEllipseRHT
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


path = "D:\\1LAB"
original_image = cv2.imread(path + "\\1619_img.png", 0)
mask = cv2.imread(path + "\\1619_mask.png", 0)
mask_binary = np.zeros(mask.shape , dtype=bool)
mask_binary[mask==255] = True
mask_binary[mask!=255] = False

time1 = time.time()
test = FindEllipseRHT(original_image,mask_binary)
# plt.figure()
# plt.imshow(original_image)
# plt.show()
test.run(plot_mode=True,debug_mode=False)
time2 = time.time()
print("time consume: ", time2 - time1)

# angle = 0.5
# A = 2
# C= 3
# AXIS_MAT = np.array([[np.sin(angle)**2 , np.cos(angle)**2], [np.cos(angle)**2, np.sin(angle)**2]])
# AXIS_MAT_ANS = np.array([A, C])
# np.linalg.solve(AXIS_MAT,AXIS_MAT_ANS)
