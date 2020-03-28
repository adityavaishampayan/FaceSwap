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

        # Warp triangles
        points1 = np.float32(points1)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points1, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)

        img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area
