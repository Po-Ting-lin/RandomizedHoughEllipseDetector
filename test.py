from randomizedHoughEllipseDetection import FindEllipseRHT
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread(r"phase.png", 0)
mask = cv2.imread(r"mask.png", 0)
mask_binary = np.zeros(mask.shape, dtype=bool)
mask_binary[mask == 255] = True
mask_binary[mask != 255] = False

time1 = time.time()
find_ellipse_RHT = FindEllipseRHT(iters=1000)
result = find_ellipse_RHT.run(original_image, mask_binary)
time2 = time.time()
print("time consume: ", time2 - time1)



p = int(result.p)
q = int(result.q)
a = int(result.a)
b = int(result.b)
angle = result.angle
result = original_image.copy()
result = cv2.ellipse(result, (p, q), (a, b), angle * 180 / np.pi, 0, 360, color=255, thickness=1)
plt.figure()
plt.title("Hough ellipse detector")
plt.imshow(result, cmap='jet', vmax=255, vmin=0)
plt.show()