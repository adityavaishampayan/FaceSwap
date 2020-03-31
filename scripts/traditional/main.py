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

# @file    main.py
# @Author  Aditya Vaishampayan (adityavaishampayan)
# @copyright  MIT
# @brief a wrapper file for calling traditional cv methods

from scripts.traditional.TPS.thin_plate_splines import thin_plate_spline
from scripts.traditional.Warping.triangulation import triangulation

import numpy as np
import cv2


def conventional_method(img1, img2, points1, points2, method):
    """

    Args:
        img1: image 1
        img2: image 2
        points1: facial landmark points for face 1
        points2: facial landmark points for face 2
        method: thin plate spline or triangulation

    Returns: face swapped output

    """

    warped_image1 = np.copy(img2)

    convex_hull2 = []
    convex_hull1 = []
    cloning_hull = []

    convexhull_index = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(0, len(convexhull_index)):
        convex_hull2.append(points2[int(convexhull_index[i])])
        convex_hull1.append(points1[int(convexhull_index[i])])

    # we perform affine transformation if method == affine or if method = inv_warp we perform inverse warping
    if method == "affine" or method == "inv_warp":
        warped_image1 = triangulation(img1, img2, warped_image1, convex_hull1, convex_hull2, method)

    # we perform thin plate spline method is mode == tps
    elif method == "tps":
        warped_image1 = thin_plate_spline(img1, warped_image1, points1, points2, convex_hull2)

    for i in range(0, len(convex_hull2)):
        cloning_hull.append((convex_hull2[i][0], convex_hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(cloning_hull), (255, 255, 255))
    (x, y, w, h) = cv2.boundingRect(np.float32([convex_hull2]))
    center = (x + int(x / 2), y + int(y / 2))

    # to perform a seamless cloning operation
    seamless_clone = cv2.seamlessClone(np.uint8(warped_image1), img2, mask, center, cv2.NORMAL_CLONE)

    return seamless_clone
