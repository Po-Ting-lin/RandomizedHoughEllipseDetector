import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.feature import canny
import pandas as pd
import time


class FindEllipseRHT(object):
    """ find a ellipse by randomized Hough Transform"""
    def __init__(self, phase_img, mask):
        # phase image
        self.origin = self.assert_img(phase_img)
        self.mask = mask

        # canny edge operator
        self.edge = None
        self.edge_pixels = None

        # settings
        self.max_iter = 800
        self.major_bound = [100, 250]
        self.minor_bound = [80, 250]
        self.flattening_bound = 0.8

        # plot
        self.black = None

        # accumulator
        self.accumulator = []
        self.score_threshold = 7

    def assert_img(self, img):
        try:
            assert len(img.shape) == 2
        except AttributeError:
            raise AttributeError("img is not a numpy array!")
        return img

    def assert_edge_pixels(self, edge_pixels):
        if len(edge_pixels) == 0:
            raise AssertionError("no edge!")

    def point_out_of_image(self, point):
        """ point X, Y"""
        if point[0] < 0 or point[0] >= self.edge.shape[1] or point[1] < 0 or point[1] >= self.edge.shape[0]:
            return True
        else:
            return False

    def ellipse_out_of_image(self, pad_img, pad_width):
        if np.sum(pad_img[:pad_width, :]) + np.sum(pad_img[-pad_width:, :]) + np.sum(pad_img[:, :pad_width]) + np.sum(pad_img[:, -pad_width:]) > 0:
            return True
        else:
            return False

    def canny_edge_detector(self, img):
        edged_image = canny(img, sigma=3.5, low_threshold=25, high_threshold=30,mask = self.mask)
        edge = np.zeros(edged_image.shape, dtype=np.uint8)
        edge[edged_image == True] = 255
        edge[edged_image == False] = 0
        plt.figure()
        plt.imshow(edge)
        plt.show()
        return edge

    def run(self, debug_mode=False, plot_mode=False):
        # seed
        random.seed((time.time()*100) % 50)

        # canny
        self.edge = self.canny_edge_detector(self.origin)


        # find coordinates of edge
        edge_pixels = np.array(np.where(self.edge == 255)).T
        self.assert_edge_pixels(edge_pixels)
        self.edge_pixels = [p for p in edge_pixels]

        # demo
        self.black = np.zeros(self.edge.shape, dtype=np.uint8)  # demo

        # determine the number of iteration
        if len(self.edge_pixels) > 100:
            self.max_iter = len(self.edge_pixels) * 5

        if debug_mode:
            candidate = 0

        for count, i in enumerate(range(self.max_iter)):
            # find nice ellipse
            p1, p2, p3 = self.randomly_pick_point()
            point_package = [p1, p2, p3]

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
                semi_axisx, semi_axisy, angle = self.find_semi_axis(point_package, center)
            except np.linalg.LinAlgError as err:
                print(err)
                continue

            # assert is ellipse
            if (semi_axisx is None) and (semi_axisy is None):
                continue

            # assert diameter
            if not self.assert_diameter(semi_axisx, semi_axisy):
                continue

            # plot
            center_plot = (int(round(center[0])), int(round(center[1])))
            if semi_axisx > semi_axisy:
                axis_plot = (int(round(semi_axisx)), int(round(semi_axisy)))
            else:
                axis_plot = (int(round(semi_axisy)), int(round(semi_axisx)))

            # bound
            pad_width = 2
            pad_black = np.zeros((self.edge.shape[0] + pad_width, self.edge.shape[1] + pad_width), dtype=np.uint8)
            pad_black = cv2.ellipse(pad_black, (center_plot[0] + pad_width, center_plot[1] + pad_width), axis_plot, angle * 180 / np.pi, 0, 360, color=255, thickness=1)
            if self.ellipse_out_of_image(pad_black, pad_width):
                continue

            if plot_mode:
                self.black = cv2.ellipse(self.black, center_plot, axis_plot, angle*180/np.pi, 0, 360, color=255, thickness=1)

            # accumulate
            similar_idx = self.is_similar(center[0], center[1], semi_axisx, semi_axisy, angle)
            if similar_idx == -1:
                score = 1
                # print("append!!")
                self.accumulator.append([center[0], center[1], semi_axisx, semi_axisy, angle, score])
            else:
                # score += 1
                self.accumulator[similar_idx][-1] += 1
                # average weights
                w = self.accumulator[similar_idx][-1]
                self.accumulator[similar_idx][0] = self.average_weight(self.accumulator[similar_idx][0], center[0], w)
                self.accumulator[similar_idx][1] = self.average_weight(self.accumulator[similar_idx][1], center[1], w)
                self.accumulator[similar_idx][2] = self.average_weight(self.accumulator[similar_idx][2], semi_axisx, w)
                self.accumulator[similar_idx][3] = self.average_weight(self.accumulator[similar_idx][3], semi_axisy, w)
                self.accumulator[similar_idx][4] = self.average_weight(self.accumulator[similar_idx][4], angle, w)

            # plot candidate
            if debug_mode:
                candidate += 1
                if candidate == 10:
                    print("axis_plot", 2*axis_plot[0], 2*axis_plot[1])
                    self.black = cv2.ellipse(self.black, center_plot, axis_plot, angle * 180 / np.pi, 0, 360, color=255, thickness=1)
                    self.black[int(center[1]), int(center[0])] = 230
                    self.find_center(point_package, plot_mode=True)
                    for point in point_package:
                        self.black[point[1], point[0]] = 80
                    break

        if plot_mode or debug_mode:
            plt.figure()  # demo
            plt.title("nice ellipse")
            plt.imshow(self.black, cmap='jet')
            plt.show()

        # sort ellipse candidates
        self.accumulator = np.array(self.accumulator)
        df = pd.DataFrame(data=self.accumulator, columns=['x', 'y', 'axis1', 'axis2', 'angle', 'score'])
        self.accumulator = df.sort_values(by=['score'], ascending=False)

        # select the ellipses with the highest score
        best = np.squeeze(np.array(self.accumulator.iloc[0]))
        p, q, a, b = list(map(int, np.around(best[:4])))
        if plot_mode:
            self.plot_best(p, q, a, b, best[-2])
        print("score: ", best[-1])
        self.inner_outer_phase(p, q, a, b, best[-2])

    def plot_best(self, p, q, a, b, angle):
        # plot best ellipse
        result = np.zeros(self.edge.shape, dtype=np.uint8)
        if a > b :
            result = cv2.ellipse(result, (p, q), (a, b), angle * 180 / np.pi, 0, 360, color=255, thickness=1)

        else :
            result = cv2.ellipse(result, (p, q), (b, a), angle * 180 / np.pi, 0, 360, color=255, thickness=1)


        # # fig, ax = plt.subplots(1,2,2)
        # ax[0].imshow(self.origin, cmap='jet', vmax=255, vmin=0)
        # ax[0].set_title("phase image")
        crop = self.origin.copy()
        crop[self.edge == 255] = 0  # blue
        crop[result == 255] = 230  # red
        # ax[1].imshow(crop, cmap='jet', vmax=255, vmin=0)
        # ax[1].set_title("Hough ellipse detector")
        # fig.show()
        print(crop)
        plt.figure()
        plt.imshow(crop)
        plt.show()


    def inner_outer_phase(self, p, q, a, b, angle):
        ellipse_label = np.zeros(self.origin.shape, dtype=np.uint8)
        if a > b:
            ellipse_label = cv2.ellipse(ellipse_label, (p, q), (a, b), angle * 180 / np.pi, 0, 360, color=255, thickness=-1)
        else:
            ellipse_label = cv2.ellipse(ellipse_label, (p, q), (b, a), angle * 180 / np.pi, 0, 360, color=255, thickness=-1)
        inner_list = self.origin[ellipse_label == 255]
        outer_list = self.origin[ellipse_label != 255]
        inner_mean = np.mean(inner_list)
        inner_std = np.std(inner_list)
        outer_mean = np.mean(outer_list)
        outer_std = np.std(outer_list)
        print("inner outer std: ", inner_std, outer_std)
        print("phase ratio: ", round(inner_mean/outer_mean, 3))

    def randomly_pick_point(self):
        ran = random.sample(self.edge_pixels, 3)
        return (ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])

    def find_center(self, pt, plot_mode=False):
        size = 7
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
            if plot_mode:
                self.black = self.plot_line(self.black, m, c)

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
            if plot_mode:
                self.black = self.plot_line(self.black, slope, intercept)

        # find center
        coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]])
        dependent_variable = np.array([-intercept_arr[0], -intercept_arr[1]])
        center = np.linalg.solve(coef_matrix, dependent_variable)
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
        dependent_variable = np.array([1,1,1])
        A, B, C = np.linalg.solve(coef_matrix, dependent_variable)



        # major = np.sqrt(np.cos(2*angle) / (a - (a + c) * np.pow(np.sin(angle), 2)));
        # new_minor_axis = major / np.sqrt((a + c) * np.pow(major,2) - 1)

        if self.assert_valid_ellipse(A, B, C):
            angle = self.calculate_rotation_angle_v2(A, B, C)
            # assert np.isnan(angle)
            # angle = self.calculate_angle(A, B, C)
            AXIS_MAT = np.array([[np.sin(angle) ** 2, np.cos(angle) ** 2], [np.cos(angle) ** 2, np.sin(angle) ** 2]])
            AXIS_MAT_ANS = np.array([A, C])
            X , Y = np.linalg.solve(AXIS_MAT, AXIS_MAT_ANS)
            major = 1/np.sqrt(max(X,Y))
            # major = np.sqrt(np.cos(2*angle)/(A-(A+C)*np.sin(angle)**2))
            minor = 1/np.sqrt(min(X,Y))
            # minor = major/np.sqrt((A+C)*major**2-1)
            # A, B, C = np.sqrt(np.abs(np.reciprocal([A, B, C])))
            # print(angle)

            return major, minor, angle
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
                # angle_dist = (e[4] - angle)**2
                angle_dist = (abs(e[4] - angle))
                # print(area_dist, center_dist, angle_dist)
                laxis_dist = abs(max(axis1,axis2)-max(e[2],e[3]))
                saxis_dist = abs(min(axis1,axis2)-min(e[2],e[3]))
                if (laxis_dist < 5) and (center_dist < 5) and ( angle_dist < 0.1745) and(saxis_dist < 10):
                    return idx
        return similar_idx

    def calculate_rotation_angle(self, a, b, c):
        if b == 0 and a < c:
            # 0 degree
            angle = 0
        elif b == 0 and c < a:
            # 90 degree
            angle = np.pi/2
        elif b != 0:
            # tilt
            angle = np.arctan((c-a-np.sqrt((a - c)**2+b**2))/b)
        else:
            # circle, no angle
            angle = 0

        if angle < 0:
            angle += np.pi
        elif angle > np.pi:
            angle -= np.pi
        return angle

    def calculate_rotation_angle_v2(self, a, b, c):
        if a == c:
            angle = 0
        else:
            angle = 0.5*np.arctan((2*b)/(a-c))

        if a > c:
            if b < 0:
                angle = angle-(-0.5*np.pi)
            elif b > 0:
                angle = angle-(0.5*np.pi)
        # if not np.isnan(angle):
        #     pass
        # else:
        print(angle,a, b, c)
        return angle

    def assert_valid_ellipse(self, a, b, c):
        if a*c - b**2 > 0:
            return True
        else:
            return False

    def assert_diameter(self, semi_axis_x, semi_axis_y):
        if semi_axis_x > semi_axis_y:
            major, minor = semi_axis_x, semi_axis_y
        else:
            major, minor = semi_axis_y, semi_axis_x
        # diameter
        if (self.major_bound[0] < 2*major < self.major_bound[1]) and (self.minor_bound[0] < 2*minor < self.minor_bound[1]):
            # Flattening
            flattening = (major - minor) / major
            if flattening < self.flattening_bound:
                return True
        return True

    def average_weight(self, old, now, score):
        return (old * score + now) / (score+1)

    def plot_line(self, img, m, c):
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

        img = cv2.line(img, solution[0], solution[1], color=50, thickness=1)
        return img

