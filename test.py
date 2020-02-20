from randomizedHoughEllipseDetection import FindEllipseRHT
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import *

original_image = cv2.imread(IMAGE_PATH, 0)
mask = cv2.imread(MASK_PATH, 0)
mask_binary = np.zeros(mask.shape, dtype=bool)
mask_binary[mask == 255] = True
mask_binary[mask != 255] = False

time1 = time.time()
test = FindEllipseRHT(iters=1000, plot_mode=False)
test.run(original_image, mask_binary)
time2 = time.time()
print("time consume: ", time2 - time1)
