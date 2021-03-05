import cv2
import time
import numpy as np
import random
from skimage.feature import canny


class RHTException(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return self.message if self.message else "some thing wrong!"


class Candidate(object):
    def __init__(self, p, q, a, b, angle, info):
        self.p = p
        self.q = q
        self.a = a
        self.b = b
        self.angle = angle
        self.score = 1
        self.info = info

    def average(self, candidate):
        self.score += 1
        self.p = self.average_weight(self.p, candidate.p, self.score)
        self.q = self.average_weight(self.q, candidate.q, self.score)
        self.a = self.average_weight(self.a, candidate.a, self.score)
        self.b = self.average_weight(self.b, candidate.b, self.score)
        self.angle = self.average_weight(self.angle, candidate.angle, self.score)

    def is_similar(self, candidate):
        area_dist = np.abs((np.pi * self.a * self.b - np.pi * candidate.a * candidate.b))
        center_dist = np.sqrt((self.p - candidate.p) ** 2 + (self.q - candidate.q) ** 2)
        angle_dist = (abs(self.angle - candidate.angle))
        if candidate.angle >= 0:
            angle180 = candidate.angle - np.pi
        else:
            angle180 = candidate.angle + np.pi
        angle_dist180 = (abs(self.angle - angle180))
        angle_final = min(angle_dist, angle_dist180)

        # axis dist
        major_axis_dist = abs(max(candidate.a, candidate.b) - max(self.a, self.b))
        minor_axis_dist = abs(min(candidate.a, candidate.b) - min(self.a, self.b))
        if (major_axis_dist < self.info.SimilarMajorAxisDist) and (minor_axis_dist < self.info.SimilarMinorAxisDist) and (center_dist < self.info.SimilarCenterDist) and (angle_final < self.info.SimilarAngleDist):
            return True
        else:
            return False

    @staticmethod
    def average_weight(old, now, score):
        return (old * score + now) / (score + 1)

    def __str__(self):
        text = ""
        text += str(np.round(self.p)) + ", "
        text += str(np.round(self.q)) + ", "
        text += str(np.round(self.a)) + ", "
        text += str(np.round(self.b)) + ", "
        text += str(np.round(self.angle, 3)) + ", "
        text += str(np.round(self.score))
        return text


class Accumulator(object):
    def __init__(self):
        self.accumulator = []
        self.score_map = []

    def get_best_candidate(self):
        index = int(np.argmax(self.score_map))
        print("best: ", str(self.accumulator[index]))
        return self.accumulator[index]

    def evaluate_candidate(self, new_candidate):
        index = self.__get_similar_index(new_candidate)
        if index == -1:
            self.__add(new_candidate)
        else:
            self.__merge(index, new_candidate)

    def __add(self, candidate):
        self.accumulator.append(candidate)
        self.score_map.append(candidate.score)

    def __merge(self, index, candidate):
        self.accumulator[index].average(candidate)
        self.score_map[index] += 1

    def __get_similar_index(self, new_candidate):
        similar_idx = -1
        if len(self.accumulator) > 0:
            for idx, candidate in enumerate(self.accumulator):
                if candidate.is_similar(new_candidate):
                    return idx
        return similar_idx

    def __str__(self):
        text = ""
        for candidate in self.accumulator:
            if candidate.score > 1:
                text += str(candidate)
                text += "\n"
        return text


class FindEllipseRHTInfo(object):
    MaxIter = 1000
    MajorAxisBound = [60, 250]
    MinorAxisBound = [60, 250]
    MaxFlattening = 0.8
    CannySigma = 3.5
    CannyT1 = 25
    CannyT2 = 30
    SimilarCenterDist = 5
    SimilarMajorAxisDist = 10
    SimilarMinorAxisDist = 10
    SimilarAngleDist = np.pi / 18


class FindEllipseRHT(object):
    def __init__(self, info):
        # image
        self.origin = None
        self.mask = None

        # canny edge operator
        self.edge = None
        self.edge_pixels = None

        # settings
        self.info = info
        self.line_fitting_area = 7

        # accumulator
        self.accumulator = Accumulator()

    def run(self, phase_img, mask):
        self.assert_img(phase_img)
        self.assert_img(mask)
        self.origin = phase_img
        self.mask = mask
        self.edge = np.zeros(self.origin.shape, dtype=np.uint8)

        # seed
        random.seed((time.time()*100) % 50)

        # canny
        self.canny_edge_detector()

        # find coordinates of edge
        edge_pixels = np.array(np.where(self.edge == 255)).T

        # no edge
        if self.is_no_edge_pixel(edge_pixels):
            raise RHTException("no edge!!")
        else:
            self.edge_pixels = [p for p in edge_pixels]

        # determine the number of iteration
        if len(self.edge_pixels) > 100:
            self.info.MaxIter = len(self.edge_pixels) * 10

        self.find_candidate()

        # select the ellipses with the highest score
        best_candidate = self.accumulator.get_best_candidate()
        # print(self.accumulator)
        return best_candidate

    def find_candidate(self):
        for i in range(self.info.MaxIter):
            # randomly pick 3 points
            point_package = self.randomly_pick_point()

            # find center
            try:
                center = self.find_center(point_package)
            except np.linalg.LinAlgError:  # Singular matrix
                continue
            except RHTException:
                continue

            # find axis
            try:
                semi_major, semi_minor, angle = self.find_semi_axis(point_package, center)
            except np.linalg.LinAlgError:  # Singular matrix
                continue
            except RHTException:
                continue

            # plot
            center = (int(round(center[0])), int(round(center[1])))
            axis = (int(round(semi_major)), int(round(semi_minor)))

            # out of mask?
            if self.is_ellipse_out_of_mask(center, axis, angle):
                continue

            # accumulate
            candidate = Candidate(center[0], center[1], semi_major, semi_minor, angle, self.info)
            self.accumulator.evaluate_candidate(candidate)

    def canny_edge_detector(self):
        edged_image = canny(self.origin,
                            sigma=self.info.CannySigma,
                            low_threshold=self.info.CannyT1,
                            high_threshold=self.info.CannyT2,
                            mask=self.mask)
        self.edge[edged_image == True] = 255
        self.edge[edged_image == False] = 0

    def randomly_pick_point(self):
        ran = random.sample(self.edge_pixels, 3)
        return [(ran[0][1], ran[0][0]), (ran[1][1], ran[1][0]), (ran[2][1], ran[2][0])]

    def find_center(self, pt):
        m, c = 0, 0
        m_arr = []
        c_arr = []

        # pt[0] is p1; pt[1] is p2; pt[2] is p3;
        for i in range(len(pt)):
            # find tangent line
            xstart = pt[i][0] - self.line_fitting_area//2
            xend = pt[i][0] + self.line_fitting_area//2 + 1
            ystart = pt[i][1] - self.line_fitting_area//2
            yend = pt[i][1] + self.line_fitting_area//2 + 1
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

            slope_arr.append(slope)
            intercept_arr.append(intercept)

        # find center
        coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]])
        dependent_variable = np.array([-intercept_arr[0], -intercept_arr[1]])
        center = np.linalg.solve(coef_matrix, dependent_variable)

        # assert center
        self.point_out_of_image(center)
        return center

    def find_semi_axis(self, pt, center):
        # shift to origin
        npt = [(p[0] - center[0], p[1] - center[1]) for p in pt]

        # semi axis
        x1, y1, x2, y2, x3, y3 = np.array(npt).flatten()
        coef_matrix = np.array([[x1**2, 2*x1*y1, y1**2], [x2**2, 2*x2*y2, y2**2], [x3**2, 2*x3*y3, y3**2]])
        dependent_variable = np.array([1, 1, 1])
        A, B, C = np.linalg.solve(coef_matrix, dependent_variable)

        if A*C - B**2 > 0:
            angle = self.calculate_rotation_angle(A, B, C)
            axis_coef = np.array([[np.sin(angle)**2, np.cos(angle)**2], [np.cos(angle)**2, np.sin(angle)**2]])
            axis_ans = np.array([A, C])
            a, b = np.linalg.solve(axis_coef, axis_ans)
        else:
            raise RHTException

        if a > 0 and b > 0:
            major = 1/np.sqrt(min(a, b))
            minor = 1/np.sqrt(max(a, b))
            if self.assert_diameter(major, minor):
                return major, minor, angle
        raise RHTException

    def assert_diameter(self, major, minor):
        if (self.info.MajorAxisBound[0] < 2 * major < self.info.MajorAxisBound[1]) and (self.info.MinorAxisBound[0] < 2 * minor < self.info.MinorAxisBound[1]):
            flattening = (major - minor) / major
            if flattening < self.info.MaxFlattening:
                return True
        return False

    def point_out_of_image(self, point):
        if point[0] < 0 or point[0] >= self.edge.shape[1] or point[1] < 0 or point[1] >= self.edge.shape[0]:
            raise RHTException("center is out of image!")

    def is_ellipse_out_of_mask(self, center, axis, angle):
        e, out_of_mask = np.zeros_like(self.origin), np.zeros_like(self.origin)
        e = cv2.ellipse(e, center, axis, angle * 180 / np.pi, 0, 360, color=255, thickness=1)
        out_of_mask[(e == 255) & (self.mask == False)] = 255
        return True if np.sum(out_of_mask) > 0 else False

    @staticmethod
    def calculate_rotation_angle(a, b, c):
        if a == c:
            angle = 0
        else:
            angle = 0.5*np.arctan((2*b)/(a-c))

        if a > c:
            if b < 0:
                angle += 0.5*np.pi  # +90 deg
            elif b > 0:
                angle -= 0.5*np.pi  # -90 deg
        return angle

    @staticmethod
    def assert_img(img):
        assert len(img.shape) == 2, "this img is not 2D image"
        assert type(img).__module__ == np.__name__, "this img is not numpy array"

    @staticmethod
    def is_no_edge_pixel(edge_pixels):
        return True if len(edge_pixels) < 15 else False





