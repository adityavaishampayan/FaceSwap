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

# @file    triangulation.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief file for performing inverse warping on delaunay triangles

import numpy as np
from scipy.interpolate import interp2d
import cv2


def triangulationWarping(src, srcTri, dstTri, size, epsilon=0.1):
    """
    this function performs inverse warping using barycentric coordinates
    Args:
        src: source image
        srcTri: source triangle obtained after delaunay triangulation
        dstTri: destination triangle having same indices as source triangle is second face
        size: size of the rectangle
        epsilon: a multiplication factor

    Returns: triangle one inverse warped on triangle 2

    """
    dst_tri = dstTri
    src_tri = srcTri

    # coordinates of the bounding rectangle
    x, y, w, h = cv2.boundingRect(np.float32([dst_tri]))

    # obtain the height and width of the rectangle
    rect_left = x
    rect_right = x + w
    rect_top = y
    rect_bottom = y + h

    # obtaining the destination matrix
    matrix_dst = np.linalg.inv([[dst_tri[0][0], dst_tri[1][0], dst_tri[2][0]],
                               [dst_tri[0][1], dst_tri[1][1], dst_tri[2][1]],
                                [1, 1, 1]])

    grid = np.mgrid[rect_left:rect_right, rect_top:rect_bottom].reshape(2, -1)
    # grid 2xN
    grid = np.vstack((grid, np.ones((1, grid.shape[1]))))
    # grid 3xN
    barycentric_coords = np.dot(matrix_dst, grid)

    dst_tri = []
    b = np.all(barycentric_coords > -epsilon, axis=0)
    a = np.all(barycentric_coords < 1 + epsilon, axis=0)
    for i in range(len(a)):
        dst_tri.append(a[i] and b[i])
    dst_y = []
    dst_x = []
    for i in range(len(dst_tri)):
        if dst_tri[i]:
            dst_y.append(i % h)
            dst_x.append(i / h)

    barycentric_coords = barycentric_coords[:, np.all(-epsilon < barycentric_coords, axis=0)]
    barycentric_coords = barycentric_coords[:, np.all(barycentric_coords < 1 + epsilon, axis=0)]

    src_matrix = np.matrix([[src_tri[0][0], src_tri[1][0], src_tri[2][0]],
                            [src_tri[0][1], src_tri[1][1], src_tri[2][1]],
                            [1, 1, 1]])

    # matrix multiplication of source matrix and barycentric coordinates
    pts = np.matmul(src_matrix, barycentric_coords)

    # converting values to homogenous coordinates
    xA = pts[0, :] / pts[2, :]
    yA = pts[1, :] / pts[2, :]

    dst = np.zeros((size[1], size[0], 3), np.uint8)

    #  copy back the value of the pixel at (xA,yA) to the target location.
    #  Using scipy.interpolate.interp2d to perform this operation.
    i = 0
    for x, y in zip(xA.flat, yA.flat):
        y_values = np.linspace(0, src.shape[0], num=src.shape[0], endpoint=False)
        x_values = np.linspace(0, src.shape[1], num=src.shape[1], endpoint=False)

        g = src[:, :, 1]
        fg = interp2d(x_values, y_values, g, kind='cubic')
        green = fg(x, y)[0]

        b = src[:, :, 0]
        fb = interp2d(x_values, y_values, b, kind='cubic')
        blue = fb(x, y)[0]

        r = src[:, :, 2]
        fr = interp2d(x_values, y_values, r, kind='cubic')
        red = fr(x, y)[0]

        dst[dst_y[i], dst_x[i]] = (blue, green, red)
        i = i + 1

    return dst
