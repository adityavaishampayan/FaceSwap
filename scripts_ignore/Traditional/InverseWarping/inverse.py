# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2020 Aditya Vaishampayan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# @file    inverse.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for inverse warping method

import sys

# noinspection PyBroadException
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except BaseException:
    pass

import cv2
import numpy as np
from scipy.interpolate import interp2d

def triangulationWarping(src, srcTri, dstTri, size, epsilon=0.1):
    t = dstTri
    s = srcTri

    #(x1, y1, w1, h1)
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([t]))

    xleft = x1
    xright = x1 + w1
    ytop = y1
    ybottom = y1 + h1

    dst_matrix = np.linalg.inv([[t[0][0], t[1][0], t[2][0]],
                                [t[0][1], t[1][1], t[2][1]],
                                [1, 1, 1]])

    grid = np.mgrid[xleft:xright, ytop:ybottom].reshape(2, -1)


    # grid 2xN
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))

    print(grid.shape)

    # grid 3xN
    barycoords = np.dot(dst_matrix, grid)


    t = []
    b = np.all(barycoords > -epsilon, axis=0)
    a = np.all(barycoords < 1 + epsilon, axis=0)
    for i in range(len(a)):
        t.append(a[i] and b[i])
    dst_y = []
    dst_x = []
    for i in range(len(t)):
        if (t[i]):
            dst_y.append(i % h1)
            dst_x.append(i / h1)

    barycoords = barycoords[:, np.all(-epsilon < barycoords, axis=0)]
    barycoords = barycoords[:, np.all(barycoords < 1 + epsilon, axis=0)]

    src_matrix = np.matrix([[s[0][0], s[1][0], s[2][0]], [s[0][1], s[1][1], s[2][1]], [1, 1, 1]])
    pts = np.matmul(src_matrix, barycoords)

    xA = pts[0, :] / pts[2, :]
    yA = pts[1, :] / pts[2, :]

    dst = np.zeros((size[1], size[0], 3), np.uint8)
    print("dst: ", dst.shape)

    i = 0
    for x, y in zip(xA.flat, yA.flat):
        xs = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)
        ys = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)

        b = src[:, :, 0]
        fb = interp2d(xs, ys, b, kind='cubic')

        g = src[:, :, 1]
        fg = interp2d(xs, ys, g, kind='cubic')

        r = src[:, :, 2]
        fr = interp2d(xs, ys, r, kind='cubic')

        blue = fb(x, y)[0]
        green = fg(x, y)[0]
        red = fr(x, y)[0]

        print("blue: ",blue)
        print("green: ", green)
        print("red: ", red)

        dst[dst_y[i], dst_x[i]] = (blue, green, red)
        i = i + 1

    return dst



def inv_warping(indexes_triangles, img1, img2, face1_points, face2_points, lines_space_mask, img2_new_face):

    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = face1_points[triangle_index[0]]
        tr1_pt2 = face1_points[triangle_index[1]]
        tr1_pt3 = face1_points[triangle_index[2]]
        # coordinates of triangle 1
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        # Obtaining the rectangles
        rect1 = cv2.boundingRect(triangle1)
        (x1, y1, w1, h1) = rect1
        cropped_triangle = img1[y1: y1 + h1, x1: x1 + w1]
        cropped_tr1_mask = np.zeros((h1, w1), np.uint8)

        # Offset points by left top corner of the respective rectangles
        points1 = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1],
                            [tr1_pt2[0] - x1, tr1_pt2[1] - y1],
                            [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points1, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img1, img1, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = face2_points[triangle_index[0]]
        tr2_pt2 = face2_points[triangle_index[1]]
        tr2_pt3 = face2_points[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x2, y2, w2, h2) = rect2

        cropped_tr2_mask = np.zeros((h2, w2), np.uint8)

        # Offset points by left top corner of the respective rectangles
        points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                            [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                            [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # t1rect = points1
        # t2rect = points2
        # size = (w2, h2)
        # img1Rect = cropped_triangle


        warped_triangle = triangulationWarping(cropped_triangle, points1, points2, (w2, h2))

        # # Warp triangles
        # points1 = np.float32(points1)
        # points2 = np.float32(points2)
        # M = cv2.getAffineTransform(points1, points2)
        # warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)

        img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area
