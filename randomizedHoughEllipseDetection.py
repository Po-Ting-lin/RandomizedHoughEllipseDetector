import cv2
import numpy as np
import random
import math
import time
from matplotlib import pyplot as plt
from skimage.feature import canny
import pandas as pd


class FindEllipseRHT(object):
    """ find a ellipse by randomized Hough Transform"""
    def __init__(self, phase_img):
        self.assert_img(phase_img)
        self.edge = self.canny_edge_detector(phase_img)
        self.edge_pixels = None

        # settings
        self.max_iter = 700
        self.major_bound = [40, 110]
        self.minor_bound = [40, 110]
        self.flattening_bound = 0.5

        # accumulator
        self.accumulator = []

    def assert_img(self, img):
        try:
            assert len(img.shape) == 2
        except AttributeError:
            raise AttributeError("img is not a numpy array!")

    def canny_edge_detector(self, img):
        edged_image = canny(img, sigma=3.5, low_threshold=25, high_threshold=50)
        edge = np.zeros(edged_image.shape, dtype=np.uint8)
        edge[edged_image == True] = 255
        edge[edged_image == False] = 0
        return edge

    def run(self):
        random.seed(41)
        edge_pixels = np.array(np.where(self.edge == 255)).T
        self.edge_pixels = [p for p in edge_pixels]
        black = np.zeros(self.edge.shape, dtype=np.uint8)  # demo

        for count, i in enumerate(range(self.max_iter)):
            # current iteration
            # print(count)
            # find potential ellipse
            p1, p2, p3 = self.randomly_pick_point()
            point_package = [p1, p2, p3]
            # print(p1, p2, p3)
            for point in point_package:
                black[point[1], point[0]] = 255

            # find center
            try:
                center = self.find_center(point_package)
            except np.linalg.LinAlgError as err:
                print(err)
                continue

            # assert center is reasonable
            if self.point_out_of_image(center):
                continue

            # find axis
            try:
                semi_axis1, semi_axis2, angle = self.find_semi_axis(point_package, center)
            except np.linalg.LinAlgError as err:
                print(err)
                continue

            # assert is ellipse
            if (semi_axis1 is None) and (semi_axis2 is None):
                continue

            # assert diameter
            if not self.assert_diameter(semi_axis1, semi_axis2):
                continue

            # plot
            center_plot = (int(round(center[0])), int(round(center[1])))
            axis_plot = (int(round(semi_axis1)), int(round(semi_axis2)))

            # bound
            pad_width = 2
            pad_black = np.zeros((self.edge.shape[0] + pad_width, self.edge.shape[1] + pad_width), dtype=np.uint8)
            pad_black = cv2.ellipse(pad_black, (center_plot[0] + pad_width, center_plot[1] + pad_width), axis_plot, angle * 180 / np.pi, 0, 360, color=255, thickness=1)
            if self.ellipse_out_of_image(pad_black, pad_width):
                continue

            # demo
            black = cv2.ellipse(black, center_plot, axis_plot, angle*180/np.pi, 0, 360, color=255, thickness=1)

            # accumulate
            similar_idx = self.is_similar(center[0], center[1], semi_axis1, semi_axis2, angle)
            if similar_idx == -1:
                score = 1
                # print("append!!")
                self.accumulator.append([center[0], center[1], semi_axis1, semi_axis2, angle, score])
            else:
                # score += 1
                self.accumulator[similar_idx][-1] += 1
                # average weights
                w = self.accumulator[similar_idx][-1]
                self.accumulator[similar_idx][0] = self.average_weight(self.accumulator[similar_idx][0], center[0], w)
                self.accumulator[similar_idx][1] = self.average_weight(self.accumulator[similar_idx][1], center[1], w)
                self.accumulator[similar_idx][2] = self.average_weight(self.accumulator[similar_idx][2], semi_axis1, w)
                self.accumulator[similar_idx][3] = self.average_weight(self.accumulator[similar_idx][3], semi_axis2, w)
                self.accumulator[similar_idx][4] = self.average_weight(self.accumulator[similar_idx][4], angle, w)

        plt.figure()  # demo
        plt.title("nice ellipse")
        plt.imshow(black)
        plt.show()

        # sort ellipse candidates
        self.accumulator = np.array(self.accumulator)
        df = pd.DataFrame(data=self.accumulator, columns=['x', 'y', 'axis1', 'axis2', 'angle', 'score'])
        self.accumulator = df.sort_values(by=['score'], ascending=False)

        # select the ellipse with the best score
        best = np.squeeze(np.array(self.accumulator.iloc[1]))
        best = list(map(int, np.around(best)))

        # plot best ellipse
        result = np.zeros(self.edge.shape, dtype=np.uint8)
        result = cv2.ellipse(result, (best[0], best[1]), (best[2], best[3]), best[4] * 180 / np.pi, 0, 360, color=255, thickness=1)
        plt.figure()  # demo
        plt.title("best ellipse")
        plt.imshow(result)
        plt.show()

    def randomly_pick_point(self):
        ran = random.sample(self.edge_pixels, 3)
        return (ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])

    def find_center(self, pt):
        size = 5
        m, c = 0, 0
        m_arr = []
        c_arr = []

        # pt[0] is p1; pt[1] is p2; pt[2] is p3;
        for i in range(len(pt)):
            # find tangent line
            xstart = pt[i][0] - size//2
            xend = pt[i][0] + size//2 + 1
            ystart = pt[i][1] - size//2
            yend = pt[i][1] + size//2 + 1
            crop = self.edge[ystart: yend, xstart: xend].T
            proximal_point = np.array(np.where(crop == 255)).T
            proximal_point[:, 0] += xstart
            proximal_point[:, 1] += ystart

            # fit straight line
            A = np.vstack([proximal_point[:, 0], np.ones(len(proximal_point[:, 0]))]).T
            m, c = np.linalg.lstsq(A, proximal_point[:, 1], rcond=None)[0]
            m_arr.append(m)
            c_arr.append(c)

        # find intersection
        slope_arr = []
        intercept_arr = []
        for i, j in zip([0, 1], [1, 2]):
            # intersection
            coef_matrix = np.array([[m_arr[i], -1], [m_arr[j], -1]])
            dependent_variable = np.array([-c_arr[i], -c_arr[j]])
            t12 = np.linalg.solve(coef_matrix, dependent_variable)
            # middle point
            m1 = ((pt[i][0] + pt[j][0])/2, (pt[i][1] + pt[j][1])/2)
            # bisector
            slope = (m1[1] - t12[1]) / (m1[0] - t12[0])
            intercept = (m1[0]*t12[1] - t12[0]*m1[1]) / (m1[0] - t12[0])
            # self.plot_line(slope, intercept)
            slope_arr.append(slope)
            intercept_arr.append(intercept)

        # find center
        coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]])
        dependent_variable = np.array([-intercept_arr[0], -intercept_arr[1]])
        center = np.linalg.solve(coef_matrix, dependent_variable)
        # print(center)
        return center

    def find_semi_axis(self, pt, center):
        # shift to origin
        npt = []
        for p in pt:
            npt.append((p[0] - center[0], p[1] - center[1]))
        # semi axis
        x1 = npt[0][0]
        y1 = npt[0][1]
        x2 = npt[1][0]
        y2 = npt[1][1]
        x3 = npt[2][0]
        y3 = npt[2][1]
        coef_matrix = np.array([[x1**2, 2*x1*y1, y1**2], [x2**2, 2*x2*y2, y2**2], [x3**2, 2*x3*y3, y3**2]])
        dependent_variable = np.array([1, 1, 1])
        sol = np.linalg.solve(coef_matrix, dependent_variable)
        angle = self.calculate_rotation_angle(sol[0], sol[1], sol[2])
        if self.assert_valid_ellipse(sol[0], sol[1], sol[2]):
            sol = np.sqrt(np.abs(np.reciprocal(sol)))
            semi_axis1 = sol[0]
            semi_axis2 = sol[2]
            return semi_axis1, semi_axis2, angle
        else:
            return None, None, None

    def is_similar(self, p, q, axis1, axis2, angle):
        similar_idx = -1
        if self.accumulator is not None:
            for idx, e in enumerate(self.accumulator):
                # area dist
                area_dist = np.abs((np.pi*e[2]*e[3] - np.pi * axis1 * axis2))
                # center dist
                center_dist = np.sqrt((e[0] - p)**2 + (e[1] - q)**2)
                # angle dist
                angle_dist = (e[4] - angle)**2
                # print(area_dist, center_dist, angle_dist)
                if (area_dist < 500) and (center_dist < 10) and (angle_dist < 10):
                    return idx
        return similar_idx

    def calculate_rotation_angle(self, a, b, c):
        if b == 0 and a < c:
            # 0 degree
            angle = 0
        elif b == 0 and c > a:
            # 90 degree
            angle = np.pi/2
        elif b != 0:
            # tilt
            angle = np.arctan((c-a-np.sqrt((a - c)**2+b**2))/b) + np.pi/2
        else:
            # circle, no angle
            angle = 0
        return angle

    def ellipse_out_of_image(self, pad_img, pad_width):
        if np.sum(pad_img[:pad_width, :]) + np.sum(pad_img[-pad_width:, :]) + np.sum(pad_img[:, :pad_width]) + np.sum(pad_img[:, -pad_width:]) > 0:
            return True
        else:
            return False

    def assert_valid_ellipse(self, a, b, c):
        if a*c - b**2 > 0:
            return True
        else:
            return False

    def assert_diameter(self, semi_major_axis, semi_minor_axis):
        # diameter
        if (self.major_bound[0] < 2*semi_major_axis < self.major_bound[1]) and (self.minor_bound[0] < 2*semi_minor_axis < self.minor_bound[1]):
            # Flattening
            flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis
            if -self.flattening_bound < flattening < self.flattening_bound:
                return True
        return False

    def average_weight(self, old, now, score):
        return (old * score + now) / (score+1)

    def point_out_of_image(self, point):
        """ point X, Y"""
        if point[0] < 0 or point[0] >= self.edge.shape[1] or point[1] < 0 or point[1] >= self.edge.shape[0]:
            return True
        else:
            return False

    def plot_line(self, m, c):
        if m == 0:
            return
        solution = []
        # left bound
        x = 0
        if 0 <= m*x+c < self.edge.shape[0]:
            solution.append((x, int(round(m * x + c))))

        # right bound
        x = self.edge.shape[1] - 1
        if 0 <= m*x+c < self.edge.shape[0]:
            solution.append((x, int(round(m * x + c))))

        # upper bound
        y = 0
        if 0 <= (y - c)/m < self.edge.shape[1]:
            solution.append((int(round((y - c)/m)), y))

        # lower bound
        y = self.edge.shape[0] - 1
        if 0 <= (y - c)/m < self.edge.shape[1]:
            solution.append((int(round((y - c)/m)), y))

        # print("solution: ", solution)
        canvas = np.zeros(self.edge.shape, dtype=np.uint8)
        canvas = cv2.line(canvas, solution[0], solution[1], color=255, thickness=1)
        plt.figure()
        plt.title("line")
        plt.imshow(canvas)
        plt.show()


path = r"C:\Users\BT\Desktop\kaggle\RPE_crop_image"
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




