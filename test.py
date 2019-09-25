from randomizedHoughEllipseDetection import FindEllipseRHT
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


path = "D:\\1LAB"
original_image = cv2.imread(path + "\\1704_img.png", 0)
mask = cv2.imread(path + "\\1704_mask.png", 0)
mask_binary = np.zeros(mask.shape , dtype=bool)
mask_binary[mask==255] = True
mask_binary[mask!=255] = False

time1 = time.time()
test = FindEllipseRHT(original_image,mask_binary)
# plt.figure()
# plt.title("origin")
# plt.imshow()
# plt.show()
test.run(plot_mode=True)
time2 = time.time()
print("time consume: ", time2 - time1)

