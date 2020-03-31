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

# @file    thin_plate_splines.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for obtaining the face warped after applying thin plate splines

import numpy as np
import cv2
import math
from scripts.traditional.TPS.estimating_params import param_estimation
from scripts.traditional.TPS.U_r import U


def thin_plate_spline(image_1, image_2, facial_landmark_pts1, facial_landmark_pts2, convex_hull2):
    """
    a function for swapping two faces by performing thin plate splines
    Args:
        image_1: image 1
        image_2: image 2
        facial_landmark_pts1: facial landmark points of the first face
        facial_landmark_pts2: facial landmark points of the second face
        convex_hull2: onvex hull obtained from the points of the second face

    Returns: face swapped image

    """
    # obtaining the number of facial landmarks
    p = len(facial_landmark_pts1)

    # converting facial landmarks from face 1 to a numpy array
    facial_landmark_pts1 = np.asarray(facial_landmark_pts1)

    # converting facial landmarks from face 2 to a numpy array
    facial_landmark_pts2 = np.asarray(facial_landmark_pts2)

    # obtaining a bounding rectangle
    rect = cv2.boundingRect(np.float32([facial_landmark_pts2]))

    # the x, y and height and the width
    rect_x, rect_y, rect_w, rect_h = rect

    # obtaining a mask of size of the height and width of the rectangle
    mask = np.zeros((rect_h, rect_w, 3), dtype = np.float32)

    points2_t = []

    convex_hull_length = len(convex_hull2)
    for i in range(convex_hull_length):
        points2_t.append(((convex_hull2[i][0] - rect_x), (convex_hull2[i][1] - rect_y)))

    cv2.fillConvexPoly(mask, np.int32(points2_t), (1.0, 1.0, 1.0), 16, 0)
    
    parameters_y = param_estimation(facial_landmark_pts2, facial_landmark_pts1[:, 1])
    parameters_x = param_estimation(facial_landmark_pts2, facial_landmark_pts1[:, 0])

    ax_x = parameters_x[p]
    ax_y = parameters_y[p]

    ay_x = parameters_x[p+1]
    ay_y = parameters_y[p+1]

    a1_x = parameters_x[p+2]
    a1_y = parameters_y[p+2]

    warped_img = np.copy(mask)

    warp_img_height = warped_img.shape[1]
    warp_img_width = warped_img.shape[0]

    for i in range(warp_img_height):
        for j in range(warp_img_width):
            l = 0
            t = 0
            m = j + rect_y
            n = i + rect_x

            b = [n, m]

            for k in range(p):
                a = facial_landmark_pts2[k, :]
                norm_ab = U(np.linalg.norm((a-b)))
                t = t + parameters_x[k] * norm_ab
                l = l + parameters_y[k] * norm_ab

            y = int(a1_y + ax_y*n + ay_y*m + l)
            x = int(a1_x + ax_x*n + ay_x*m + t)

            x = min(max(x, 0), image_1.shape[1] - 1)
            y = min(max(y, 0), image_1.shape[0] - 1)

            warped_img[j, i] = image_1[y, x, :]

    warped_img = warped_img * mask

    image_2[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = image_2[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] * \
                                                              ((1.0, 1.0, 1.0) - mask)
    image_2[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = image_2[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] + \
                                                              warped_img

    return image_2
