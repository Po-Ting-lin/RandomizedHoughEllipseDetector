from randomizedHoughEllipseDetection import FindEllipseRHT
import time
import cv2

original_image = cv2.imread(path + "\\3894_img.png", 0)

import time
time1 = time.time()
test = FindEllipseRHT(original_image)
plt.figure()
plt.title("origin")
plt.imshow(test.edge)
plt.show()
test.run()
time2 = time.time()
print("time consume: ", time2 - time1)
