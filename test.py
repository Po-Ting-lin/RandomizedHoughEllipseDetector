from randomizedHoughEllipseDetection import EllipseDetector, EllipseDetectorInfo
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    original_image = cv2.imread(r"phase.png", 0)
    mask = cv2.imread(r"mask.png", 0)
    mask_binary = np.zeros(mask.shape, dtype=bool)
    mask_binary[mask == 255] = True
    mask_binary[mask != 255] = False

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax[0].set_title("original")
    ax[0].imshow(original_image, cmap='jet', vmax=255, vmin=0)
    ax[0].axis("off")
    ax[1].set_title("mask")
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis("off")

    info = EllipseDetectorInfo()
    info.MaxIter = 5000
    info.MaxIter = 1000
    info.MajorAxisBound = [60, 250]
    info.MinorAxisBound = [60, 250]
    info.MaxFlattening = 0.8
    info.CannySigma = 3.5
    info.CannyT1 = 25
    info.CannyT2 = 30
    info.SimilarCenterDist = 5
    info.SimilarMajorAxisDist = 10
    info.SimilarMinorAxisDist = 10
    info.SimilarAngleDist = np.pi / 18

    time1 = time.time()
    obj = EllipseDetector(info)
    result = obj.run(original_image, mask_binary)
    time2 = time.time()
    print("time consume: {:.2f} s".format(time2 - time1))

    p = int(result.p)
    q = int(result.q)
    a = int(result.a)
    b = int(result.b)
    angle = result.angle
    result = original_image.copy()
    result = cv2.ellipse(result, (p, q), (a, b), angle * 180 / np.pi, 0, 360, color=255, thickness=1)
    ax[2].set_title("result")
    ax[2].imshow(result, cmap='jet', vmax=255, vmin=0)
    ax[2].axis("off")
    plt.tight_layout()
    plt.show()
