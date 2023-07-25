import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing

from scipy.ndimage import binary_fill_holes as imfill
from skimage.draw import line_aa
from glob import glob

MAXINT = 2147483647

ROOT = '/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/MDE/kitti/'
FILES = 'kitti_eigen_train.txt'
FILES2 = 'kitti_eigen_test.txt'

class LinPers:
    """
    Linear Perspective class. Converts RGB images (3xHxW) into HxW linear perspective maps.
    """

    def __init__(self):
        """
        self.BLUR: (odd) int, amount of Gaussian blur (for x and y) to apply to images
        self.CANNY1: int, Canny edge detector lower threshold argument
        self.CANNY2: int, Canny edge detector upper threshold argument
        self.RHO: int, Hough line detector rho argument
        self.THETA: float, Hough line detector theta argument
        self.REJECT: float, minimum degree that a line should deviate from 90 and 0
        self.TOP: int, max number of lines (descending by length) to include in the vanishing point detection
        self.PARAMS: [(int,int,int)], list of Hough line detector arguments threshold, min_line_len and max_line_gap
        self.RADIUS: float, max distance that a line can have to the vanishing point during filtering
        self.MERGE: float, threshold for the distance to merge two points that lie on the same edge
        self.edges: np.ndarray, edges of the image to be processed
        self.w: int, width of the image
        self.c: int, maximum index of the horizontal axis
        self.h: int, height of the image
        self.r: int, maximum index of the vertical axis
        self.lines: [(int,int,int,int,float,float,float)], list of lines detected in an image
        self.vx: float, x-coordinate of the vanishing point
        self.vy: float, y-coordinate of the vanishing point
        self.pts: [(float, float)], list of points on the edge of the image that lie on a line to the vanishing point
        self.gradient: np.ndarray, resulting gradient
        """
        self.BLUR = 3
        self.CANNY1 = 40
        self.CANNY2 = 200
        self.RHO = 1
        self.THETA = np.pi / 180
        self.PARAMS = [
            (20, 30, 10),
            (10, 20, 10),
            (25, 50, 30),
            (20, 10, 30),
            (50, 5, 10)
        ]
        self.REJECT = 3
        self.TOP = 50
        self.MAXINT = 2147483647
        self.RADIUS = 15
        self.MERGE = 20
        self.IMPORTANT = .5
        self.NOT_IMPORTANT = .2

        self.edges = None
        self.w = -1
        self.c = -1
        self.h = -1
        self.r = -1
        self.lines = []
        self.vx = -1
        self.vy = -1
        self.pts = []
        self.gradient = None
        self.max_cnt = 1

    def generate(self, images, save=True):
        for i, img_url in enumerate(images):
            img = cv2.imread(ROOT + 'input/' + img_url, cv2.IMREAD_UNCHANGED)
            self.set_image(img, reset_size=True)
            self.find_lines()
            self.vanishing_pnt()
            self.filter_lines()
            self.edge_pts()
            self.draw_gradient()
            if save:
                res = (self.gradient * 65535).astype('uint16')
                res_url = ROOT + 'L/' + img_url
                cv2.imwrite(res_url, res)

    def set_image(self, img, reset_size=True):
        """
        Determines the edges of an image and sets it for later processing
        :param img: np.ndarray, BGR input image
        :param reset_size: boolean, determines if the previous width and height values need to be replaced
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if reset_size:
            self.w = gray.shape[1]
            self.c = self.w - 1
            self.h = gray.shape[0]
            self.r = self.h - 1
            self.vx = self.w // 2
            self.vy = self.h // 2
        blurred = cv2.GaussianBlur(gray, (self.BLUR, self.BLUR), 0)
        self.edges = cv2.Canny(blurred, self.CANNY1, self.CANNY2)
        self.lines = []
        self.pts = []
        self.gradient = np.zeros((self.h, self.w), dtype='float32')

    def find_lines(self):
        """
        Determines the Hough lines for multiple combinations of parameters. Finds the slope (a), and y-intercept (b) of
        the mathematical representation of the line (y=ax+b), as well as the length of the line piece and adds these as
        extra properties to the list of lines. Lines with an angle too close to 0 or 90 degrees are skipped.
        """
        for thresh, min_len, max_gap in self.PARAMS:
            hough_lines = cv2.HoughLinesP(self.edges, self.RHO, self.THETA, thresh, min_len, max_gap)
            if hough_lines is not None:
                for [[x1, y1, x2, y2]] in hough_lines:
                    if x1 == x2:
                        continue
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1
                    if self.REJECT < abs(math.degrees(math.atan(a))) < 90 - self.REJECT:
                        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5
                        self.lines.append([x1, y1, x2, y2, a, b, length])
        self.lines = sorted(self.lines, key=lambda x: x[6], reverse=True)

    def vanishing_pnt(self):
        """
        Determines the vanishing point of the image by finding the point that minimises the error. This error is the sum
        of errors for each of the (longest) lines which is based on the distance to a potential point and the square of
        the length of the line.
        """
        min_err = MAXINT
        for i, line1 in enumerate(self.lines[:self.TOP]):
            for line2 in self.lines[i + 1:self.TOP]:
                a1, b1, a2, b2 = line1[4], line1[5], line2[4], line2[5]
                if a1 == a2:
                    continue
                x0 = (b1 - b2) / (a2 - a1)
                y0 = a1 * x0 + b1
                err = 0
                for line3 in self.lines[:self.TOP]:
                    a3, b3, l3 = line3[4], line3[5], line3[6]
                    a4 = -1 / a3
                    b4 = y0 - a4 * x0
                    x = (b3 - b4) / (a4 - a3)
                    y = a4 * x + b4
                    err += l3 ** 2 * abs((y - y0) ** 2 + (x - x0) ** 2)
                if min_err > err:
                    min_err = err
                    self.vx, self.vy = x0, y0

    def filter_lines(self):
        """
        Filter the lines on whether they intersect the vanishing point within a certain radius
        """
        self.lines = [(x1, y1, x2, y2) for (x1, y1, x2, y2, a, b, l) in self.lines
                      if ((a * self.vx - self.vy + b) ** 2) ** .5 / (a ** 2 + 1) ** .5 < self.RADIUS]

    def edge_pts(self):
        """
        Determines the point where a line from the vanishing point that passes through the furthest end of each line
        intersects the edge of the image.
        """
        for x1, y1, x2, y2 in self.lines:
            d1 = ((x1 - self.vx) ** 2 + (y1 - self.vy) ** 2) ** .5
            d2 = ((x2 - self.vx) ** 2 + (y2 - self.vy) ** 2) ** .5
            x, y = [(x1, y1), (x2, y2)][int(d2 > d1)]  # get the furthest point of the two
            a = MAXINT if x == self.vx else (self.vy - y) / (self.vx - x)
            b = y - a * x
            # for intersection with borders: left, right, top and bottom:
            for cx, cy in [(0, b), (self.c, a * self.c + b), (-b / a, 0), ((self.r - b) / a, self.r)]:
                if 0 <= cx < self.w and 0 <= cy < self.h:
                    d1 = ((cx - self.vx) ** 2 + (cy - self.vy) ** 2) ** .5
                    d2 = ((cx - x) ** 2 + (cy - y) ** 2) ** .5
                    if d1 > d2:  # vanishing point is further away from the border than the reference point
                        break
            self.pts.append((cx, cy, 1))
        self.sort_pts()
        self.merge_pts()
        self.filter_pts()

    def draw_gradient(self):
        """
        Creates an image of the same shape as the input image, with a gradient from the edges (lowest) to the vanishing
        point (highest), for every slice of the image. The borders of such a slice are lines from border points to the
        vanishing point.
        """
        if len(self.pts) < 4 or self.max_cnt < 2 or self.vx < 0 or self.vx > self.c or self.vy < 0 or self.vy > self.r:
            self.vx = self.w // 2
            self.vy = self.h // 2
            self.pts = [(0, 0, 10), (0, self.r, 10), (self.c, self.r, 10), (self.c, 0, 10)]
        for idx, ((x1, y1, _), (x2, y2, _)) in enumerate(zip(self.pts, self.pts[1:] + self.pts[:1])):
            mask = self.create_mask([(x1, y1), (x2, y2)])
            a1 = MAXINT if y1 == self.vy else (self.vx - x1) / (self.vy - y1)
            b1 = x1 - a1 * y1
            a2 = MAXINT if y2 == self.vy else (self.vx - x2) / (self.vy - y2)
            b2 = x2 - a2 * y2
            grad = np.fromfunction(lambda i, j: self.gradient_function(i, j, a1, b1, a2, b2),
                                   (self.h, self.w), dtype='float32')
            grad = 1 - grad / np.max(grad[mask])
            self.gradient[mask] = grad[mask]

    def sort_pts(self):
        """
        Sorts the points that lay on the edge counter-clockwise starting at (0,0).
        """
        left = sorted([(x, y, c) for (x, y, c) in self.pts if x == 0], key=lambda p: p[1])
        bottom = sorted([(x, y, c) for (x, y, c) in self.pts if y == self.r], key=lambda p: p[0])
        right = sorted([(x, y, c) for (x, y, c) in self.pts if x == self.c], key=lambda p: p[1], reverse=True)
        top = sorted([(x, y, c) for (x, y, c) in self.pts if y == 0], key=lambda p: p[0], reverse=True)
        bl_corner = [(x, y, c) for (x, y, c) in [(0, self.r, 1)] if self.vx < 0]
        br_corner = [(x, y, c) for (x, y, c) in [(self.c, self.r, 1)] if self.vx > self.c]
        self.pts = left + bl_corner + bottom + br_corner + right + top

    def merge_pts(self):
        """
        Merges points on the same edge if they lie near each other. Requires the points to be sorted
        """
        i = 0
        cnt = 1
        while i < len(self.pts) - 1:
            x1, y1, c1 = self.pts[i]
            x2, y2, c2 = self.pts[i + 1]
            if x1 != x2 and y1 != y2 or (abs(x1 - x2) > self.MERGE or abs(y1 - y2) > self.MERGE):
                cnt = 1
                i += 1
            else:  # points on same edge and close enough for merge to make sense
                cnt += 1
                if c1 + c2 > self.max_cnt:
                    self.max_cnt = c1 + c2
                if x1 == x2:
                    self.pts[i] = x1, y1 * (cnt - 1) / cnt + y2 / cnt, c1 + c2
                else:  # y1 == y2
                    self.pts[i] = x1 * (cnt - 1) / cnt + x2 / cnt, y1, c1 + c2
                self.pts.pop(i + 1)

    def filter_pts(self):
        """
        Removes border points from self.pts that result from a small number of merges,
        when it is between two border points that both result from a large number of merges.
        """
        imp_pts = []
        for i, (_, _, c) in enumerate(self.pts):
            if c / self.max_cnt > self.IMPORTANT:
                imp_pts.append(i)
        to_remove = []
        if len(imp_pts) > 1:
            for i, (x, y, c) in enumerate(self.pts):
                if i < imp_pts[0] or i > imp_pts[-1]:
                    if len(imp_pts) > 3 and c / self.max_cnt < self.NOT_IMPORTANT:
                        to_remove.append(i)
                else:
                    if c / self.max_cnt < self.NOT_IMPORTANT:
                        to_remove.append(i)
        new_pts = []
        for i, (x, y, c) in enumerate(self.pts):
            if i not in to_remove:
                new_pts.append((x, y, c))
        self.pts = new_pts

    def create_mask(self, vertices):
        """
        Creates a mask for one slice (area between two border points) of the image.
        :param vertices: list containing the two border points
        :return: The mask for the slice
        """
        mask = np.zeros((self.h, self.w), dtype=bool)
        verts = [(self.vx, self.vy)] + vertices
        i = 1
        while i < len(verts) - 1:
            x1, y1 = verts[i]
            x2, y2 = verts[i + 1]
            if x1 != x2 and y1 != y2:
                if x1 == 0 and y1 != self.r:
                    verts.insert(i + 1, (0, self.r))
                elif y1 == self.r and x1 != self.c:
                    verts.insert(i + 1, (self.c, self.r))
                elif x1 == self.c and y1 != 0:
                    verts.insert(i + 1, (self.c, 0))
                elif y1 == 0 and x1 != 0:
                    verts.insert(i + 1, (0, 0))
            i += 1
        x1, y1 = self.vanish_in_bound(*verts[1])
        x2, y2 = self.vanish_in_bound(*verts[-1])
        if x1 is None:
            x1, y1 = x2, y2
        if x2 is None:
            x2, y2 = x1, y1
        verts = [(x2, y2), (x1, y1)] + verts[1:]
        verts.append((x2, y2))
        for (x1, y1), (x2, y2) in zip(verts, verts[1:] + verts[:1]):
            rs, cs = line_aa(int(y1), int(x1), int(y2), int(x2))[:2]
            mask[rs, cs] = 1
        return imfill(mask)

    def vanish_in_bound(self, ex, ey):
        """
        Determines the point on the line from reference point (ex,ey) to the vanish point that lies within the image and
        is closest to the vanishing point. Requires that the reference point lies on an edge of the image.
        :param ex: x-coordinate of the reference point
        :param ey: y-coordinate of the reference point
        :return: x- and y-coordinate of the resulting point
        """
        if 0 <= self.vx < self.w and 0 <= self.vy < self.h:
            return self.vx, self.vy
        else:
            a = MAXINT if ex == self.vx else (self.vy - ey) / (self.vx - ex)
            b = ey - a * ex
            edges = [(x, y) for (x, y) in [(0, b), (self.c, a * self.c + b), (-b / a, 0), ((self.r - b) / a, self.r)]
                     if 0 <= x < self.w and 0 <= y < self.h and (x, y) != (ex, ey)]
            if len(edges) != 1:
                return None, None
            return edges[0]

    @staticmethod
    def gradient_function(x, y, a1, b1, a2, b2):
        """
        Calculates the value for a point (x, y) based on the distance between two lines.
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param a1: slope of the first line
        :param b1: y-intercept of the first line
        :param a2: slope of the second line
        :param b2: y-intercept of the second line
        :return: sum of the distance from point to line 1 and distance from point to line 2
        """
        return abs(a1 * x - y + b1) / (a1 ** 2 + 1) ** .5 + abs(a2 * x - y + b2) / (a2 ** 2 + 1) ** .5


def show(cv2_img):
    """
    Display image, while converting BGR to RGB, and showing grayscale with color map 'magma'
    :param cv2_img: np.ndarray, image to be shown
    """
    if len(cv2_img.shape) == 2:
        plt.imshow(cv2_img, cmap='magma')
    else:
        plt.imshow(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    plt.show()


def execute(img_urls):
    lp = LinPers()
    lp.generate(img_urls, save=True)


fls = []
with open(ROOT + FILES, 'r') as f:
    fls += [line.split()[0] for line in f.readlines()]
with open(ROOT + FILES2, 'r') as f:
    fls += [line.split()[0] for line in f.readlines()]
sub_lists = []
for i in range(len(fls) // 100 + 1):
    sub_lists.append(fls[i * 100: (i+1) * 100])

pool = multiprocessing.Pool()
pool.map(execute, sub_lists)
